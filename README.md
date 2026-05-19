# Monosemanticity — Sparse Autoencoders on Llama-3.2-1B

A self-contained reproduction of the recipe from Anthropic's
[Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity)
paper, adapted to small open-weight models. The pipeline extracts
residual-stream activations from a transformer, trains a sparse autoencoder
(SAE) on them, and generates an HTML report describing what each learned
"feature" appears to detect — including a natural-language description from an
LLM-judge and a token-level activation heat-map for the most strongly
activating sequences.

```
download.py  ──►  extract.py  ──►  train.py  ──►  infer.py  ──►  analyse.py
   │                │                │              │              │
   ▼                ▼                ▼              ▼              ▼
model +         (train, test)     SAE          feature        report.html
dataset        activations    checkpoints     tensor (F,N,S)
                                              + side arrays
```

## Quick start

```bash
# 1. Create venv + install deps
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Run the pipeline (defaults assume 1× RTX 3090 + ~1.7 TB free disk)
MONO_NUM_GPUS=1 .venv/bin/python download.py    # ~6 min  (one-time)
MONO_NUM_GPUS=1 .venv/bin/python extract.py     # ~8 min
MONO_NUM_GPUS=1 .venv/bin/python train.py       # ~22 min for 50K steps
MONO_NUM_GPUS=1 .venv/bin/python infer.py       # ~22 min  (262 GB write, HDD-bound)
              .venv/bin/python analyse.py       # ~30 s    (no GPU)

# 3. View the report
xdg-open data/analysis/report.html
```

The pipeline is restartable — every script no-ops if its output already exists.

## Pipeline

| Script        | GPU  | What it does                                                                | Inputs                       | Outputs                                                       |
|---------------|------|-----------------------------------------------------------------------------|------------------------------|---------------------------------------------------------------|
| `download.py` | no   | Snapshot model weights from HF, stream-download dataset shards.             | network                      | `data/models/<model>/`, `data/datasets/<dataset>/`            |
| `extract.py`  | yes  | Forward-pass dataset through the LM, capture layer-N residual stream.        | model + dataset              | `data/activations/<model>/layer<N>/{train,test}/`             |
| `train.py`    | yes  | Train SAE on train-split activations using the buffered loader.              | train activations            | `data/sae_checkpoints/sae_step_*.pt`                          |
| `infer.py`    | yes  | Run the latest SAE on the test set, save the full feature tensor + summary side-arrays. | latest checkpoint + test activations | `data/features/<model>/layer<N>/`                            |
| `analyse.py`  | no   | Filter and sample features, query an LLM for descriptions, render HTML.      | features dir                 | `data/analysis/report.html`                                   |

See `docs/pipeline.md` for a detailed walkthrough of each stage.

## Architecture highlights

* **SAE forward**: `f = ReLU(W_enc · (x − b_dec) + b_enc)`, `x̂ = W_dec · f + b_dec`,
  loss = `MSE(x, x̂) + λ · Σᵢ fᵢ · ‖w_dec_i‖`. Standard scaling-monosemanticity setup;
  decoder columns are *not* unit-norm during training, the per-column norm is rolled
  into the L1 weight (paper §4).
* **HDD-friendly activation loader**: training activations live in a 60+ GB memmap.
  Naive token-level shuffling triggers thousands of random seeks per batch and dies
  on a spinning disk. We instead read whole sequences in a worker thread into a RAM
  buffer (`BUFFER_SEQUENCES` ≈ 8 GB by default), shuffle *tokens* within the buffer
  once, and drain it batch-by-batch — sequential reads on disk, near-token-level
  randomisation in batches.
* **Feature-major storage** (`infer.py`): the features tensor is stored as
  `(F, N, S)` C-major fp16 (`F` = dictionary size, `N` = num test sequences,
  `S` = seq len). Each feature row is `N·S·2 B ≈ 2 MB` *contiguous* on disk, so
  `analyse.py` can read 100 features in ~200 MB. We also save small
  `max_per_seq` / `argmax_per_seq` / `fire_count` / `decoder_directions`
  side-arrays so analyse.py never has to scan the big tensor for top-k.
* **No GPU in analyse.py**: it only loads side-arrays, slices per-feature
  rows from the features memmap, calls an OpenAI-compatible chat-completions
  endpoint for descriptions, and renders HTML with token-level green
  highlighting proportional to the per-token feature activation.
* **Feature selection**: by default `analyse.py` picks features that
  (a) fire on between `1e-4` and `5e-2` of all test tokens (Goldilocks band),
  (b) fire on at least `ANALYSIS_TOP_K` distinct sequences, and
  (c) are mutually different in decoder direction (greedy farthest-point on cosine).
  See `docs/configuration.md` for the knobs.

## Configuration

Every tunable lives in `constants.py` and is overridable via a `MONO_*`
environment variable. The most common knobs:

| Env var                             | Default                           | What it controls                                                |
|-------------------------------------|-----------------------------------|------------------------------------------------------------------|
| `MONO_MODEL_NAME`                   | `meta-llama/Llama-3.2-1B`         | Target language model.                                           |
| `MONO_LAYER_INDEX`                  | `8`                               | Residual-stream layer to extract.                                |
| `MONO_NUM_EXTRACT_TOKENS_TRAIN`     | `20_000_000`                      | Train-split activation budget.                                   |
| `MONO_NUM_EXTRACT_TOKENS_TEST`      | `1_000_000`                       | Test-split activation budget (used by infer + analyse).          |
| `MONO_EXPANSION_FACTOR`             | `64`                              | Dictionary size = `d_model × expansion`.                         |
| `MONO_NUM_TRAINING_STEPS`           | `200_000`                         | SAE training steps.                                              |
| `MONO_BATCH_SIZE`                   | `4096`                            | Tokens per training step.                                        |
| `MONO_BUFFER_SEQUENCES`             | `2048`                            | Activation-loader RAM buffer.                                    |
| `MONO_L1_COEFF`                     | `5.0`                             | Sparsity penalty.                                                |
| `MONO_LR`                           | `5e-5`                            | Adam learning rate.                                              |
| `MONO_NUM_GPUS`                     | `2`                               | GPUs used (use `1` if only one is free).                         |
| `MONO_INFER_FEATURE_BLOCK`          | `512`                             | Feature-axis chunk size during inference.                        |
| `MONO_FEATURE_DTYPE`                | `float16`                         | Stored feature dtype.                                            |
| `MONO_ANALYSIS_NUM_FEATURES`        | `100`                             | Features in the report.                                          |
| `MONO_ANALYSIS_TOP_K`               | `20`                              | Top examples per feature.                                        |
| `MONO_ANALYSIS_MIN_FIRE_FRAC`       | `1e-4`                            | Filter: minimum firing rate.                                     |
| `MONO_ANALYSIS_MAX_FIRE_FRAC`       | `5e-2`                            | Filter: maximum firing rate.                                     |
| `MONO_ANALYSIS_DIVERSE_SELECTION`   | `1`                               | Use farthest-point selection on decoder cosine.                  |
| `MONO_LLM_API_BASE_URL`             | `http://127.0.0.1:8000`           | OpenAI-compatible endpoint for descriptions.                     |
| `MONO_LLM_API_MODEL`                | `empyrean`                        | Model name on that endpoint.                                     |

Full list in `docs/configuration.md`.

## Hardware notes

Defaults are calibrated for a single 24 GB RTX 3090 + ~256 GB free disk. The big
disk-bound number is **262 GB** for the feature tensor (262 GB ≈ `dict_size × N × S × 2 B`
for the defaults). Drop `MONO_EXPANSION_FACTOR` or `MONO_NUM_EXTRACT_TOKENS_TEST`
proportionally if you don't have the room.

Memory peaks:
- `train.py`: ~8 GB GPU (Adam state + SAE) + ~8 GB RAM (activation buffer).
- `infer.py`: ~12 GB GPU + ~20 GB RSS at steady state (Linux page cache backing the memmap).
- `analyse.py`: ~2 GB RAM (side arrays).

## Repository layout

```
.
├── README.md           — this file
├── AGENTS.md           — guidance for AI coding agents
├── constants.py        — single source of truth for all tunables
├── download.py         — model + dataset download
├── extract.py          — residual-stream extraction
├── train.py            — SAE training (model + buffered loader)
├── infer.py            — feature tensor generation
├── analyse.py          — HTML report generation
├── requirements.txt
├── paper.html          — original Anthropic paper, archived
├── docs/
│   ├── architecture.md     — system design + data flow
│   ├── pipeline.md         — per-script walkthroughs
│   ├── data-formats.md     — on-disk file specs
│   ├── configuration.md    — full env-var reference
│   └── performance.md      — bottleneck analysis & tuning
└── data/                   — generated outputs (gitignored)
    ├── models/
    ├── datasets/
    ├── activations/
    ├── sae_checkpoints/
    ├── features/
    └── analysis/
```

## References

* [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity) — Anthropic, 2024
* [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features) — Anthropic, 2023
* [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model) — Elhage et al., 2022
