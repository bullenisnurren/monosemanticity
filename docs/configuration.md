# Configuration reference

Every tunable lives in `constants.py` and is overridable via a `MONO_*`
environment variable. Listed here in the order they appear in the file.

Defaults are calibrated for **1× RTX 3090 (24 GB) + ~1.7 TB free disk +
~120 GB RAM**.

---

## Paths

| Constant       | Derivation                                   | Description                    |
|----------------|----------------------------------------------|--------------------------------|
| `PROJECT_ROOT` | Directory containing `constants.py`.         | Repo root.                     |
| `DATA_DIR`     | `PROJECT_ROOT / "data"`                      | Top-level outputs directory (typically a symlink to a large external disk in real use). |

## Model & dataset

| Env var                       | Default                          | Description                                    |
|-------------------------------|----------------------------------|------------------------------------------------|
| `MONO_MODEL_NAME`             | `meta-llama/Llama-3.2-1B`        | HF model ID.                                   |
| `MONO_DATASET_NAME`           | `monology/pile-uncopyrighted`    | HF dataset ID.                                 |
| `MONO_DATASET_SPLIT`          | `train`                          | HF split to stream.                            |
| `MONO_DATASET_TEXT_FIELD`     | `text`                           | Field of each row holding the text.            |

Derived paths (not env-overridable):
- `MODEL_DIR = DATA_DIR / "models" / <slug(MODEL_NAME)>`
- `DATASET_DIR = DATA_DIR / "datasets" / <slug(DATASET_NAME)>`

## Activation extraction

| Env var                          | Default      | Description                                          |
|----------------------------------|--------------|------------------------------------------------------|
| `MONO_LAYER_INDEX`               | `8`          | 0-indexed residual-stream layer to extract.          |
| `MONO_SEQ_LEN`                   | `512`        | Sequence length used for the forward pass.           |
| `MONO_NUM_EXTRACT_TOKENS_TRAIN`  | `20_000_000` | Train-split activation budget.                       |
| `MONO_NUM_EXTRACT_TOKENS_TEST`   | `1_000_000`  | Test-split activation budget.                        |

Derived paths:
- `ACTIVATIONS_DIR = DATA_DIR / "activations" / <slug(MODEL_NAME)> / "layer{LAYER_INDEX}"`
- `ACTIVATIONS_TRAIN_DIR = ACTIVATIONS_DIR / "train"`
- `ACTIVATIONS_TEST_DIR  = ACTIVATIONS_DIR / "test"`

## SAE architecture

| Env var                  | Default | Description                                           |
|--------------------------|---------|-------------------------------------------------------|
| `MONO_EXPANSION_FACTOR`  | `64`    | `dict_size = d_model × expansion_factor`. With `d_model=2048` this gives `dict_size = 131 072`. |

## Training hyper-parameters

| Env var                  | Default     | Description                                                          |
|--------------------------|-------------|----------------------------------------------------------------------|
| `MONO_NUM_TRAINING_STEPS`| `200_000`   | Total optimiser steps.                                               |
| `MONO_BATCH_SIZE`        | `4096`      | Tokens per training step.                                            |
| `MONO_BUFFER_SEQUENCES`  | `2048`      | Activation-loader RAM buffer (in sequences ≈ `~8 GB` at defaults).   |
| `MONO_L1_COEFF`          | `5.0`       | Sparsity penalty.                                                    |
| `MONO_L1_WARMUP_FRAC`    | `0.05`      | Fraction of training steps used to linearly warm L1 up to `L1_COEFF`.|
| `MONO_LR`                | `5e-5`      | Adam learning rate.                                                  |
| `MONO_LR_DECAY_FRAC`     | `0.20`      | Fraction of training steps over which LR linearly decays to 0.       |
| `MONO_ADAM_BETA1`        | `0.9`       | Adam β₁.                                                             |
| `MONO_ADAM_BETA2`        | `0.999`     | Adam β₂.                                                             |
| `MONO_GRAD_CLIP_NORM`    | `1.0`       | Global gradient-norm cap.                                            |
| `MONO_DECODER_INIT_NORM` | `0.1`       | Initial norm of each decoder column.                                 |

## Hardware

| Env var          | Default | Description                                                  |
|------------------|---------|--------------------------------------------------------------|
| `MONO_NUM_GPUS`  | `2`     | GPUs to use (cuda:0 … cuda:{NUM_GPUS-1}). Set to `1` if you only have one free GPU. |

## Checkpointing & logging

| Env var                | Default  | Description                                |
|------------------------|----------|--------------------------------------------|
| `MONO_CHECKPOINT_EVERY`| `10_000` | Save a checkpoint every N optimiser steps. |
| `MONO_LOG_EVERY`       | `10`     | Update the tqdm postfix every N steps.     |

`CHECKPOINT_DIR = DATA_DIR / "sae_checkpoints"`.

## Inference (`infer.py`)

| Env var                     | Default     | Description                                                                  |
|-----------------------------|-------------|------------------------------------------------------------------------------|
| `MONO_FEATURE_DTYPE`        | `float16`   | dtype of stored features. `float32` doubles disk usage with no real precision gain. |
| `MONO_INFER_FEATURE_BLOCK`  | `512`       | Number of features computed and written per loop iteration. Larger = bigger contiguous writes & higher GPU mem peak. |

`FEATURES_DIR = DATA_DIR / "features" / <slug(MODEL_NAME)> / "layer{LAYER_INDEX}"`.

## Analysis (`analyse.py`)

| Env var                            | Default               | Description                                                                                       |
|------------------------------------|-----------------------|---------------------------------------------------------------------------------------------------|
| `MONO_ANALYSIS_NUM_FEATURES`       | `100`                 | Features sampled for the report.                                                                  |
| `MONO_ANALYSIS_TOP_K`              | `20`                  | Top-activating sequences shown per feature.                                                       |
| `MONO_ANALYSIS_SEED`               | `0`                   | RNG seed for reproducible sampling.                                                               |
| `MONO_ANALYSIS_MIN_FIRE_FRAC`      | `1e-4`                | Goldilocks band — minimum firing rate (fraction of test tokens).                                  |
| `MONO_ANALYSIS_MAX_FIRE_FRAC`      | `5e-2`                | Goldilocks band — maximum firing rate.                                                            |
| `MONO_ANALYSIS_MIN_DISTINCT_SEQUENCES` | `ANALYSIS_TOP_K` | Minimum number of distinct test sequences a feature must fire in to be eligible.                  |
| `MONO_ANALYSIS_DIVERSE_SELECTION`  | `1`                   | If true: greedy farthest-point selection on decoder cosine. Otherwise uniform random over candidates. |

`ANALYSIS_DIR = DATA_DIR / "analysis"`.

## LLM-based descriptions (`analyse.py`)

| Env var                  | Default                    | Description                                                |
|--------------------------|----------------------------|------------------------------------------------------------|
| `MONO_LLM_API_BASE_URL`  | `http://127.0.0.1:8000`    | OpenAI-compatible API base URL.                            |
| `MONO_LLM_API_KEY`       | `EMPTY`                    | Bearer token for the LLM API (most local servers ignore).  |
| `MONO_LLM_API_MODEL`     | `empyrean`                 | Model name passed in the request payload.                  |
| `MONO_LLM_API_TIMEOUT`   | `60.0`                     | Per-request timeout (seconds).                             |
| `MONO_LLM_NUM_EXAMPLES`  | `10`                       | Top-k examples included in each prompt (≤ `ANALYSIS_TOP_K`).|
| `MONO_LLM_EXAMPLE_CHARS` | `240`                      | Max characters per example (centred around the peak token).|

## Tips for tuning

- **GPU memory tight in training?** Drop `MONO_BATCH_SIZE` or
  `MONO_EXPANSION_FACTOR`. The SAE's parameter count is `2 · d · F`, so
  halving `EXPANSION_FACTOR` halves the model.
- **Disk space tight?** Drop `MONO_NUM_EXTRACT_TOKENS_TEST` (the test split
  directly controls the size of `features.npy`). Halving it from 1 M → 500 K
  saves ~130 GB.
- **Report features look uninteresting?** Tighten the Goldilocks band, e.g.
  `MONO_ANALYSIS_MIN_FIRE_FRAC=5e-4 MONO_ANALYSIS_MAX_FIRE_FRAC=1e-2`.
- **Report features look like duplicates?** Make sure
  `MONO_ANALYSIS_DIVERSE_SELECTION=1` (the default).
