"""Report local data manifests and optional base-checkpoint metadata."""

import argparse
import json
import os
from pathlib import Path
import random
import sys


def read_object(path):
    with path.open(encoding="utf-8") as stream:
        try:
            result = json.load(stream)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return result


def estimate_tokens(train_dir, train_docs, sample_docs, tokenizer, parquet):
    """Sample uniformly over prepared rows and count tokens with one BOS per row."""
    if not isinstance(train_docs, int) or train_docs <= 0:
        raise ValueError("manifest lacks a positive training document count")
    rows = sorted(random.Random(42).sample(range(train_docs), min(sample_docs, train_docs)))
    sample_index = 0
    offset = token_count = 0
    for path in sorted(train_dir.glob("*.parquet")):
        with parquet.ParquetFile(path) as part:
            for group in range(part.num_row_groups):
                end = offset + part.metadata.row_group(group).num_rows
                if sample_index < len(rows) and rows[sample_index] < end:
                    texts = part.read_row_group(group, columns=["text"]).column("text")
                    while sample_index < len(rows) and rows[sample_index] < end:
                        text = texts[rows[sample_index] - offset].as_py()
                        token_count += len(tokenizer.encode(text, prepend=tokenizer.get_bos_token_id()))
                        sample_index += 1
                offset = end
    if offset != train_docs or sample_index != len(rows):
        raise ValueError(f"Parquet row count ({offset}) differs from manifest ({train_docs})")
    return round(token_count * train_docs / len(rows)), len(rows)


def report(args):
    base = args.base_dir.expanduser()
    if not base.is_dir():
        raise ValueError(f"Artifact directory does not exist: {base}. Pass --base-dir for an existing cache.")

    meta = read_object(args.checkpoint.expanduser()) if args.checkpoint else {}
    mixture = meta.get("mixture", {})
    if not isinstance(mixture, dict):
        mixture = {}  # SFT metadata does not provide pretraining mixture weights here.
    loader = meta.get("dataloader_state_dict", {})
    intake = loader.get("tokens", {})
    cursors = loader.get("sources", {})
    total_intake = sum(intake.values())
    fingerprints = {
        (kind, name): fingerprint
        for stage in meta.get("compute_ledger", {}).get("stages", [])
        for kind, sources in stage.get("data", {}).items()
        for name, fingerprint in sources.items()
    }

    data = base / "data"
    manifests = {
        ("pretrain", path.parent.name): path
        for path in data.glob("*/manifest.json")
    }
    manifests.update({("sft", path.parent.name): path for path in data.glob("sft/*/manifest.json")})
    manifests.update({("rl", path.parent.name): path for path in data.glob("rl/*/manifest.json")})
    expected = set(fingerprints) | {("pretrain", name) for name in mixture.keys() | intake.keys() | cursors.keys()}

    print(f"Artifacts: {base}")
    print(f"Checkpoint metadata: {args.checkpoint or 'unavailable (no --checkpoint supplied)'}")
    print("Intake and epochs, when present, describe the saved loader state (rank zero for multi-GPU).")
    print("Intake is measured before packing and cropping and can include prefetched documents.")
    if not manifests and not expected:
        raise ValueError(f"No built source manifests found under {data}")

    tokenizer = parquet = None
    if args.sample_docs:
        try:
            from pyarrow import parquet
            from nanochat.tokenizer import RustBPETokenizer

            tokenizer = RustBPETokenizer.from_directory(base / "tokenizer")
        except (ImportError, OSError, ValueError) as exc:
            print(f"Token sampling unavailable: {exc}", file=sys.stderr)
        if tokenizer is not None:
            print(f"Sampling tokenizer: {base / 'tokenizer'} ({tokenizer.get_vocab_size():,} entries)")

    failed = bool(args.sample_docs and tokenizer is None)
    for kind, name in sorted(manifests.keys() | expected):
        key = (kind, name)
        print(f"\n{kind}/{name}")
        manifest = {}
        if key in manifests:
            try:
                manifest = read_object(manifests[key])
            except (OSError, ValueError) as exc:
                print(f"  Manifest unreadable: {exc}", file=sys.stderr)
                failed = True
        else:
            print("  Manifest unavailable.")
        counts = manifest.get("counts", {})
        print("  Document counts: " + ", ".join(
            f"{field}={counts.get(field, 'unavailable')}"
            for field in ("raw", "empty", "filtered", "duplicates", "contaminated", "train", "val")
        ))
        actual, expected_fp = manifest.get("fingerprint"), fingerprints.get(key)
        print(f"  Manifest fingerprint: {actual or 'unavailable'}")
        print(f"  Ledger fingerprint:   {expected_fp or 'unavailable'}")
        if actual and expected_fp and actual != expected_fp:
            print("  WARNING: built source differs from the checkpoint's recorded source.")
        if kind != "pretrain":
            continue

        weight, tokens = mixture.get(name), intake.get(name)
        epoch = cursors.get(name, {}).get("epoch")
        print("  Configured intake target: " + (f"{weight:.1%}" if weight is not None else "unavailable"))
        print("  Saved intake: " + (
            f"{tokens:,} tokens ({tokens / total_intake:.2%})" if tokens is not None and total_intake else "unavailable"
        ))
        print(f"  Source epoch: {epoch if epoch is not None else 'unavailable'}")
        if epoch is not None and epoch > 1:
            print("  Reuse: the loader began another pass over this source.")
        if tokenizer is not None and manifest:
            try:
                estimate, size = estimate_tokens(
                    manifests[key].parent / "train", counts.get("train"), args.sample_docs, tokenizer, parquet
                )
                print(f"  Available train tokens: ~{estimate:,} ({size:,} sampled rows, one BOS per row, before cropping)")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                print(f"  Available train tokens: unavailable ({exc})", file=sys.stderr)
                failed = True
        else:
            print("  Available train tokens: unavailable" + (" (use --sample-docs to estimate)" if not args.sample_docs else ""))
    return int(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path(os.getenv("NANOCHAT_BASE_DIR") or "~/.cache/nanochat"),
                        help="existing cache root (defaults to NANOCHAT_BASE_DIR or ~/.cache/nanochat)")
    parser.add_argument("--checkpoint", type=Path, help="base checkpoint meta_*.json file, never a model weights file")
    parser.add_argument("--sample-docs", type=int, default=0, metavar="N",
                        help="sample up to N training documents per pretraining source using the cache's trusted tokenizer")
    args = parser.parse_args()
    if args.sample_docs < 0:
        parser.error("--sample-docs must be nonnegative")
    try:
        return report(args)
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        print(f"Could not report mixture data: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
