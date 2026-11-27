# Project 2 pretraining mixture experiments

**Question:** At equal training compute, what happens to math and Python accuracy when we replace 20% of FineWeb intake with one domain source?

## Runs and hypotheses

| Run | FineWeb / FineMath / CodeParrot | Hypothesis |
|---|---|---|
| `baseline` | 100 / 0 / 0 | Reproduce the official recipe on our hardware and establish our control score. |
| `math_20` | 80 / 20 / 0 | Educational math text improves math accuracy, especially on word problems. Check code for regressions. |
| `code_20` | 80 / 0 / 20 | Python source improves function-writing accuracy. Check whether gains transfer to Python word problems and whether math regresses. |

### Confirmed recipe

Use [the shared data config](../configs/data/project2_domain_sources.json). It declares four canonical FineWeb sample-10BT shards, five FineMath-4+ shards, and four CodeParrot-clean shards. Build them through the official pipeline. The tokenizer uses the canonical FineWeb build for every run. Use the [baseline](../configs/mixtures/baseline.json), [math](../configs/mixtures/math_20.json), and [code](../configs/mixtures/code_20.json) mixture files, respectively.

Hold fixed the official 32,768-entry tokenizer, randomly initialized d12 architecture, full attention, 1,024-token context, baseline optimizer and learning-rate schedule, 512 sequences per update, and 2,520 pretraining updates. Keep GPU count, precision, and device batch consistent across arms where possible. Run the baseline SFT mixture `gsm8k:4,mbpp:8,tinygsm:1,smoltalk:1` with its default dataset-driven stopping behavior. TinyGSM and SmolTalk are capped at 21,000 and 46,000 training conversations, respectively, during preparation. Use no RL. Check actual SFT step counts and training FLOPs in the checkpoint ledgers.

### Estimates and decisions after preparation

- **Compute:** 1.321B pretraining token slots and about 1.057e18 training FLOPs per arm if SFT takes the reported 85 steps. Budget roughly five A100-hours for the three arms, plus queue time, data preparation, and any repeats or reproduction runs.
- **Storage:** The baseline estimates about 8 GB for the four raw FineWeb shards alone. Use cluster scratch and plan for tens of GB including domain shards, prepared data, temporary build files, and checkpoints. Confirm quota, retention, network access for data staging, and CPU/GPU access to the same filesystem before building.
- **Supply:** The catalog suggests about 2.8B raw FineWeb tokens and, assuming equal shard sizes, about 780M raw FineMath tokens for five shards. It gives no token estimate for the four CodeParrot shards. Measure prepared supply with the official tokenizer. Each domain targets 20% of intake, about 264M tokens if no tokens are lost to packing, or roughly 406M if overall cropping resembles the baseline's 35% estimate.
- **Reuse:** Divide expected incoming tokens by estimated prepared training tokens for an approximate pass count. Save the first wrap step and final epoch from the training logs. Epoch 2 means the loader started a second pass. Flag unexpectedly early wraps or multiple passes and record how they affect interpretation. The saved intake counters include buffered tokens before packing and can include prefetch. Report their scope as rank zero for multi-GPU runs.

## Cluster handoff and outputs

**Michael** provides the configs, prepared manifests and fingerprints, and the prepared token estimates. **Rex** confirms GPU allocation, job wall time, and scratch capacity, then uses the commands below in the existing cluster GPU environment. The commands use one GPU. For multi-GPU, follow the [baseline runner](../runs/baseline.sh) launch pattern and use the same GPU count for pretraining and SFT so the optimizer shards can be loaded.

```bash
# Set NANOCHAT_BASE_DIR to the team's approved scratch directory first.
python -m scripts.prepare_data --config configs/data/project2_domain_sources.json
python -m scripts.tok_train

# Set arm to baseline, math_20, or code_20 for each job.
arm=baseline
tag="project2_${arm}"
python -m scripts.base_train --depth=12 --max-seq-len=1024 --window-pattern=L \
  --num-iterations=2520 --batch-size=512 --device-batch-size=32 \
  --mixture="configs/mixtures/${arm}.json" --model-tag="$tag"
python -m scripts.chat_sft --model-tag="$tag" --output-tag="$tag" \
  --mixture="gsm8k:4,mbpp:8,tinygsm:1,smoltalk:1" \
  --device-batch-size=32 --val-every=0 --save-every=25
```

Rex saves the launch arguments, revision, hardware and precision, train logs, manifests, final base metadata JSON, and SFT checkpoint tag and step with its ledger. **Tarek** evaluates the final SFT checkpoints under the same settings on all 638 validation items. He saves the summary and per-item results, reports the four subset scores, two family scores and overall score, and computes paired differences against our control. Review both the examples gained and lost. Training variation can be several points, so treat small differences cautiously. Keep any later checkpoint selection separate from the planned final-checkpoint comparison.

```bash
# Set tag and step to Rex's final SFT checkpoint for the arm.
mkdir -p results
python -m scripts.evaluate --model "sft:${tag}:${step}" --batch-size=256 \
  --output "results/${tag}.json" --save-outputs "results/${tag}_outputs.jsonl"
```

If both domains have manageable reuse and show complementary gains or a trade-off worth testing, consider a 60/20/20 mixture at equal compute. If the gains are small, prioritize a repeat of the most informative comparison. Leave time to reproduce and package the selected checkpoint before December 17, 2026.

## Read-only reporting

```bash
python -B -m scripts.report_mixture_data --base-dir "$NANOCHAT_BASE_DIR"
python -B -m scripts.report_mixture_data --base-dir "$NANOCHAT_BASE_DIR" \
  --checkpoint "$NANOCHAT_BASE_DIR/base_checkpoints/${tag}/meta_002520.json"
```

The default mode reads JSON with the Python standard library. The optional `--sample-docs 32` mode uses PyArrow and the cache's trusted local tokenizer to estimate pretraining supply from a fixed-seed sample of prepared training rows. It reads Parquet metadata and selected row groups on CPU. Each sampled document includes one beginning-of-sequence token. The estimates describe available tokens before cropping and can be noisy if lengths vary widely. Use the project environment for sampling and keep the sample small on your laptop.

To test safely with an existing smoke cache:

```bash
python -B -m scripts.report_mixture_data --base-dir ~/.cache/nanochat_smoke \
  --checkpoint ~/.cache/nanochat_smoke/base_checkpoints/smoke/meta_000100.json
```

Smoke artifacts use smaller prepared datasets and a smoke tokenizer. Any sampled totals from them describe that setup only.

## Preliminary Data Card

Package the selected model with the official `make_submission` command and the final experiment runner. It gathers the source manifests from the checkpoint lineage and generates source entries and `data_flops`. Fill in the team name, each generated `how_used` entry, and overall `notes`.

| Field | Draft |
|---|---|
| Team and selected checkpoint | [Team], [revision], [tag and step] |
| Declared data-selection FLOPs | [Generated total from manifests]. This recipe uses no model-based selection. |
| FineWeb use | Canonical four-shard build for the fixed tokenizer. Pretraining target [share], observed intake [share and scope], final source epoch [value]. |
| Domain source use | For each domain in the selected lineage, give the catalog name and shard count, target share, observed intake and scope, estimated supply and method, and final source epoch. |
| SFT source use | GSM8K train four repeats and MBPP train/prompt/validation eight repeats. One pass over capped TinyGSM and SmolTalk training splits. Add actual prepared counts and SFT steps [values]. |
| Preparation and contamination notes | Official formatting, empty-record removal, within-source exact deduplication, normalized word overlap removal against the released evaluation suite, and seeded train/held-out splits. No custom filtering. Released benchmark data used for evaluation and checkpoint selection. Reference the generated manifests for removals and fingerprints, and record any repetition or overlap limitations [observations]. |
