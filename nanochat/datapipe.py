"""
The standard data pipeline. Every source of training data goes through the same stages, so that all teams'
data is processed identically and every checkpoint can be traced back to exactly the data it was trained on.

!!! DO NOT MODIFY THIS FILE. Graders rebuild/verify data with our own copy of it. !!!
(The datasets come from the catalog in scripts/prepare_data.py; your own data selection goes into the `filter` hook.)

Stages, for every source (pretraining text or SFT conversations):
  1. read      raw records of a catalog dataset
  2. format    record -> document (pretraining: a string; SFT: a conversation {"messages": [...]})
  3. select    drop empty documents; optional selection filter `module:function`, called as
               function(document, record) -> bool (keep the document?). It cannot change the document.
  4. dedup     exact deduplication (whitespace-insensitive) within the source
  5. decontam  drop documents sharing a word n-gram (13, or 8 for short questions) with an evaluation suite
               (the released validation suite; graders additionally check against the hidden suite). The fixed RL
               problems also drop the documents of a fixed exclusion list (nanochat/rl.py), so that they are the
               same for everyone
  6. split     seeded shuffle, then a held-out validation split
  7. write     pretraining: data/<name>/{train,val}/part_*.parquet (one `text` column)
               SFT:         data/sft/<name>/{train,val}.jsonl (one conversation per line)
  8. manifest  data/<name>/manifest.json (or data/sft/<name>/manifest.json): the source spec, inputs and their
               hashes, counts after every stage, decontamination results, declared data FLOPs, a fingerprint
A source is "built" iff its manifest exists. Training scripts only accept built sources and record their
fingerprints in the compute ledger of every checkpoint.
"""

import os
import json
import glob
import shutil
import random
import hashlib
import importlib
from datetime import datetime, timezone
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq

from tasks.eval_suite import load_suite
from scripts.decontaminate import build_index, doc_hits

PIPELINE_VERSION = "1.1"
NGRAM, NGRAM_SHORT = 13, 8
ROW_GROUP_SIZE = 1024
TRAIN_PART_DOCS = 200_000 # pretraining documents per output parquet file

# -----------------------------------------------------------------------------
# Locations

def data_root():
    from nanochat.common import get_base_dir
    return os.path.join(get_base_dir(), "data")


def source_root(name, kind):
    assert kind in ("pretrain", "sft", "rl"), kind
    return os.path.join(data_root(), name) if kind == "pretrain" else os.path.join(data_root(), kind, name)


def read_manifest(name, kind):
    path = os.path.join(source_root(name, kind), "manifest.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fingerprints(names, kind):
    """{name: fingerprint} of built sources (used by the training scripts for the compute ledger)."""
    out = {}
    for name in names:
        manifest = read_manifest(name, kind)
        assert manifest is not None, (f"Data source '{name}' ({kind}) was not built by the standard pipeline. "
                                      f"Declare it in a data config and run: python -m scripts.prepare_data --config <config>")
        out[name] = manifest["fingerprint"]
    return out

# -----------------------------------------------------------------------------
# Helpers

def doc_text(doc):
    """The text of a document: a string, or all message contents of a conversation."""
    if isinstance(doc, str):
        return doc
    parts = []
    for message in doc["messages"]:
        content = message["content"]
        if isinstance(content, str):
            parts.append(content)
        else:
            parts.extend(p["text"] for p in content)
    return "\n".join(parts)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def content_sha256(path):
    """Hash of the *content* of an output file (documents), independent of the library that wrote it."""
    if not path.endswith(".parquet"):
        return sha256_file(path)
    h = hashlib.sha256()
    for text in pq.read_table(path, columns=["text"]).column("text").to_pylist():
        h.update(text.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def suite_signature(suite_dirs):
    sig = []
    for d in suite_dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            sig.append({"file": os.path.basename(path), "dir": os.path.basename(os.path.normpath(d)), "sha256": sha256_file(path)})
    return sig


def load_filter(spec):
    """'module.path:function' -> callable(document, record) -> bool (keep the document?)."""
    module, _, fn = spec.partition(":")
    return getattr(importlib.import_module(module), fn)

# decontamination runs in worker processes, each with its own copy of the n-gram index
_INDEX = None

def _init_worker(suite_dirs):
    global _INDEX
    items = [it for d in suite_dirs for it in load_suite(d)]
    _INDEX = build_index(items, NGRAM, NGRAM_SHORT)


def _flag_chunk(texts):
    return [sorted(doc_hits(t, _INDEX, NGRAM, NGRAM_SHORT)) for t in texts]

# -----------------------------------------------------------------------------
# Writing

def _write_parquet(texts, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(pa.Table.from_pydict({"text": texts}), path, row_group_size=ROW_GROUP_SIZE)


def _write_jsonl(docs, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

# -----------------------------------------------------------------------------
# The pipeline

def _doc_digest(doc):
    return hashlib.sha1(" ".join(doc_text(doc).split()).encode("utf-8")).digest()


def doc_key(doc):
    """The identity of a document: SHA-1 (hex) of its whitespace-normalized text, as used by dedup and exclusion lists."""
    return _doc_digest(doc).hex()


def exclude_signature(exclude):
    return hashlib.sha256("\n".join(sorted(exclude)).encode()).hexdigest() if exclude else None


def build_source(name, kind, spec, parts, to_doc, suite_dirs, num_workers=8, force=False, log=print, exclude=frozenset()):
    """
    Build one source. `parts` is a list of (input_info: dict, read: callable -> list of raw records);
    `to_doc(record)` formats a record into a document (or None to drop it). `exclude`: keys (doc_key) of documents
    to drop as contaminated, e.g. the RL problems that overlap the evaluation pools.
    spec keys used here: val_docs (pretraining, default 2000), val_fraction (SFT, default 0.05, at most 500),
    max_docs (cap on training documents/conversations after all stages), filter ('module:function'), seed.
    Returns the manifest.
    """
    out_dir = source_root(name, kind)
    signature = suite_signature(suite_dirs)
    excluded_sig = exclude_signature(exclude)
    old = read_manifest(name, kind)
    if old is not None and not force:
        if (old["spec"] == spec and old["pipeline_version"] == PIPELINE_VERSION and old["decontamination"]["suites"] == signature
                and old["decontamination"].get("exclude") == excluded_sig):
            log(f"[{name}] already built (fingerprint {old['fingerprint'][:12]}), skipping")
            return old
        log(f"[{name}] spec, pipeline or evaluation suite changed: rebuilding")
    if os.path.exists(out_dir):
        assert old is not None or not os.listdir(out_dir), (
            f"{out_dir} exists but was not built by the pipeline (no manifest). Move it away and rerun.")
        shutil.rmtree(out_dir)
    tmp_dir = out_dir + ".building"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    seed = int(spec.get("seed", 1337))
    max_docs = int(spec.get("max_docs", -1))
    user_filter = load_filter(spec["filter"]) if spec.get("filter") else None
    counts = {"raw": 0, "empty": 0, "filtered": 0, "duplicates": 0, "contaminated": 0, "train": 0, "val": 0}
    contaminated_examples = []
    n_excluded = 0
    seen = set()
    inputs = []
    train_files, kept_sft = [], []
    pending = [] # pretraining documents waiting to be written
    val_done = kind != "pretrain"
    chars = 0

    def flush(final=False):
        nonlocal pending
        while pending and (final or len(pending) >= TRAIN_PART_DOCS):
            chunk, pending = pending[:TRAIN_PART_DOCS], pending[TRAIN_PART_DOCS:]
            path = os.path.join(tmp_dir, "train", f"part_{len(train_files):05d}.parquet")
            _write_parquet(chunk, path)
            train_files.append(path)

    with Pool(processes=num_workers, initializer=_init_worker, initargs=(suite_dirs,)) as pool:
        for part_idx, (info, read) in enumerate(parts):
            if 0 < max_docs <= counts["train"] + len(kept_sft):
                break
            records = read()
            inputs.append(info)
            counts["raw"] += len(records)
            docs = []
            for record in records:
                doc = to_doc(record)
                if doc is None or not doc_text(doc).strip():
                    counts["empty"] += 1
                    continue
                if user_filter is not None and not user_filter(doc, record):
                    counts["filtered"] += 1
                    continue
                key = _doc_digest(doc)
                if key in seen:
                    counts["duplicates"] += 1
                    continue
                seen.add(key)
                if exclude and key.hex() in exclude:
                    counts["contaminated"] += 1
                    n_excluded += 1
                    continue
                docs.append(doc)
            del records
            texts = [doc_text(d) for d in docs]
            hits = [h for chunk in pool.map(_flag_chunk, [texts[i:i + 256] for i in range(0, len(texts), 256)]) for h in chunk]
            kept = []
            for doc, text, h in zip(docs, texts, hits):
                if h:
                    counts["contaminated"] += 1
                    if len(contaminated_examples) < 20:
                        contaminated_examples.append({"eval_items": h[:3], "text": text[:200]})
                else:
                    kept.append(doc)
            random.Random(seed + part_idx).shuffle(kept)
            if kind == "pretrain":
                if not val_done:
                    n_val = min(int(spec.get("val_docs", 2000)), len(kept) // 10)
                    _write_parquet(kept[:n_val], os.path.join(tmp_dir, "val", "part_00000.parquet"))
                    counts["val"] = n_val
                    kept = kept[n_val:]
                    val_done = True
                if max_docs > 0:
                    kept = kept[:max(0, max_docs - counts["train"])]
                counts["train"] += len(kept)
                chars += sum(len(t) for t in kept)
                pending.extend(kept)
                flush()
            else:
                kept_sft.extend(kept)
            log(f"[{name}] part {part_idx + 1}/{len(parts)}: {len(docs):,} unique documents, {sum(bool(h) for h in hits):,} contaminated")
        flush(final=True)

    if kind != "pretrain": # conversations (SFT sources, RL problems)
        random.Random(seed).shuffle(kept_sft)
        n_val = min(500, max(1, int(float(spec.get("val_fraction", 0.05)) * len(kept_sft)))) if len(kept_sft) > 20 else 0
        val, train = kept_sft[:n_val], kept_sft[n_val:]
        if max_docs > 0:
            train = train[:max_docs]
        _write_jsonl(train, os.path.join(tmp_dir, "train.jsonl"))
        train_files.append(os.path.join(tmp_dir, "train.jsonl"))
        if val:
            _write_jsonl(val, os.path.join(tmp_dir, "val.jsonl"))
        counts["train"], counts["val"] = len(train), len(val)
        chars = sum(len(doc_text(d)) for d in train)
    assert counts["train"] > 0, f"[{name}] no training documents left after the pipeline: {counts}"

    outputs = []
    for path in sorted(glob.glob(os.path.join(tmp_dir, "**", "*.*"), recursive=True)):
        outputs.append({"file": os.path.relpath(path, tmp_dir), "bytes": os.path.getsize(path), "content_sha256": content_sha256(path)})
    # the fingerprint depends only on the spec and the documents, not on file encodings (e.g. the pyarrow version)
    fingerprint = hashlib.sha256(json.dumps({"spec": spec, "version": PIPELINE_VERSION,
                                             "outputs": [(o["file"], o["content_sha256"]) for o in outputs]}, sort_keys=True).encode()).hexdigest()
    manifest = {
        "name": name,
        "kind": kind,
        "pipeline_version": PIPELINE_VERSION,
        "spec": spec,
        "inputs": inputs,
        "counts": counts,
        "train_chars": chars,
        "decontamination": {"suites": signature, "ngram": NGRAM, "ngram_short": NGRAM_SHORT,
                            "removed": counts["contaminated"], "examples": contaminated_examples,
                            **({"exclude": excluded_sig, "excluded": n_excluded} if exclude else {})},
        "data_flops": float(spec.get("data_flops", 0.0)),
        "outputs": outputs,
        "fingerprint": fingerprint,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(os.path.join(tmp_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    os.rename(tmp_dir, out_dir)
    log(f"[{name}] built: {counts} -> {out_dir} (fingerprint {fingerprint[:12]})")
    return manifest
