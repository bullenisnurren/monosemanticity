# Monosemanticity — Sparse Autoencoders on Llama-3.2-1B

Reproduction of the
[Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity)
SAE recipe applied to a small open-weight model. The pipeline extracts
residual-stream activations from a transformer, trains a sparse autoencoder
on them, and produces an HTML report describing what each learned feature
appears to detect.

```
download.py  ──►  extract.py  ──►  train.py  ──►  infer.py  ──►  analyse.py
   │                │                │              │              │
   ▼                ▼                ▼              ▼              ▼
model +         (train, test)     SAE          feature        report.html
dataset        activations    checkpoints      tensor
```

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python download.py
.venv/bin/python extract.py
.venv/bin/python train.py
.venv/bin/python infer.py
.venv/bin/python analyse.py

xdg-open data/analysis/report.html
```

Every script no-ops if its primary output is already present, so the pipeline
is fully restartable.

## Pipeline

| Script        | GPU | What it does                                                                  | Outputs                                                       |
|---------------|-----|-------------------------------------------------------------------------------|---------------------------------------------------------------|
| `download.py` | no  | Snapshot model from HuggingFace, stream-download dataset shards.              | `data/models/`, `data/datasets/`                              |
| `extract.py`  | yes | Forward-pass dataset through the LM, capture residual-stream activations.     | `data/activations/<model>/layer<N>/{train,test}/`             |
| `train.py`    | yes | Train SAE on the train-split activations.                                     | `data/sae_checkpoints/sae_step_*.pt`                          |
| `infer.py`    | yes | Run the latest SAE on the test set; save feature tensor + summary side-arrays. | `data/features/<model>/layer<N>/`                            |
| `analyse.py`  | no  | Filter/sample features, query an LLM for descriptions, render HTML.            | `data/analysis/report.html`                                   |

For per-script detail, see [`docs/pipeline.md`](docs/pipeline.md).
For the on-disk file specs, see [`docs/data-formats.md`](docs/data-formats.md).

## Configuration

Every tunable lives in `constants.py` and is overridable via a `MONO_*`
environment variable. The most common knobs:

| Env var                             | Default                           | What it controls                                                |
|-------------------------------------|-----------------------------------|------------------------------------------------------------------|
| `MONO_MODEL_NAME`                   | `meta-llama/Llama-3.2-1B`         | Target language model.                                           |
| `MONO_DATASET_NAME`                 | `monology/pile-uncopyrighted`     | Source dataset.                                                  |
| `MONO_LAYER_INDEX`                  | `8`                               | Residual-stream layer to extract.                                |
| `MONO_SEQ_LEN`                      | `512`                             | Sequence length for the forward pass.                            |
| `MONO_NUM_EXTRACT_TOKENS_TRAIN`     | `20_000_000`                      | Train-split activation budget.                                   |
| `MONO_NUM_EXTRACT_TOKENS_TEST`      | `1_000_000`                       | Test-split activation budget.                                    |
| `MONO_EXPANSION_FACTOR`             | `64`                              | Dictionary size = `d_model × expansion`.                         |
| `MONO_NUM_TRAINING_STEPS`           | `200_000`                         | SAE training steps.                                              |
| `MONO_BATCH_SIZE`                   | `4096`                            | Tokens per training step.                                        |
| `MONO_BUFFER_SEQUENCES`             | `2048`                            | Activation-loader RAM buffer.                                    |
| `MONO_L1_COEFF`                     | `5.0`                             | Sparsity penalty.                                                |
| `MONO_LR`                           | `5e-5`                            | Adam learning rate.                                              |
| `MONO_NUM_GPUS`                     | `2`                               | GPUs used (cuda:0 … cuda:{N-1}).                                 |
| `MONO_INFER_FEATURE_BLOCK`          | `512`                             | Feature-axis chunk size during inference.                        |
| `MONO_FEATURE_DTYPE`                | `float16`                         | Stored feature dtype.                                            |
| `MONO_ANALYSIS_NUM_FEATURES`        | `100`                             | Features in the report.                                          |
| `MONO_ANALYSIS_TOP_K`               | `20`                              | Top examples per feature.                                        |
| `MONO_ANALYSIS_MIN_FIRE_FRAC`       | `1e-4`                            | Min firing rate for a feature to be eligible.                    |
| `MONO_ANALYSIS_MAX_FIRE_FRAC`       | `5e-2`                            | Max firing rate.                                                 |
| `MONO_ANALYSIS_DIVERSE_SELECTION`   | `1`                               | Use farthest-point selection on decoder cosine.                  |
| `MONO_LLM_API_BASE_URL`             | `http://127.0.0.1:8000`           | OpenAI-compatible endpoint for descriptions.                     |
| `MONO_LLM_API_MODEL`                | `empyrean`                        | Model name on that endpoint.                                     |

See `constants.py` for the full list.

## Repository layout

```
.
├── README.md
├── AGENTS.md           — guidance for AI coding agents
├── constants.py        — single source of truth for all tunables
├── download.py
├── extract.py
├── train.py
├── infer.py
├── analyse.py
├── requirements.txt
├── docs/
│   ├── pipeline.md         — per-script walkthroughs
│   └── data-formats.md     — on-disk file specs
└── data/                   — generated outputs (gitignored)
```

## References

* [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity) — Anthropic, 2024
* [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features) — Anthropic, 2023
* [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model) — Elhage et al., 2022
