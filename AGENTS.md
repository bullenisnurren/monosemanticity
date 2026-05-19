# AGENTS.md — guidance for AI coding agents

## What this repo is

A 5-stage pipeline implementing the
[Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity)
SAE recipe on Llama-3.2-1B. The final artefact is an HTML report describing
what each SAE feature appears to detect, with token-level heatmaps.

```
download.py  →  extract.py  →  train.py  →  infer.py  →  analyse.py
```

Read [`README.md`](README.md) for the user-facing summary, then
[`docs/pipeline.md`](docs/pipeline.md) for per-script details and
[`docs/data-formats.md`](docs/data-formats.md) for on-disk file specs.

## Conventions

- **Configuration**: every tunable is a module-level constant in
  `constants.py`, declared with `_env("MONO_<NAME>", default, dtype)` so it
  can be overridden via env var. Don't sprinkle magic numbers around.
- **Paths**: derived from `DATA_DIR = PROJECT_ROOT / "data"`; per-model and
  per-layer artefacts are namespaced (`data/.../<model_slug>/layer<N>/`).
  `<model_slug>` is `MODEL_NAME` with `/` → `__`.
- **Style**: Python 3.10+ (`list[int]`, `str | None`), type-annotated
  signatures, docstrings on non-trivial functions, no emojis.
- **Dependencies**: pinned in `requirements.txt`. The LLM call in
  `analyse.py` uses stdlib `urllib.request`; don't introduce an
  `openai`/`httpx` dep for it.

## Architectural invariants

1. **Features tensor is stored feature-major** `(F, N, S)`. `analyse.py`
   reads it as a memmap and only ever slices `features[fid]`. Don't switch
   to `(N, S, F)` — analyse.py would have to scan the whole tensor.
2. **`analyse.py` does no GPU work and never invokes the LM or the SAE.**
   Everything it needs is precomputed by `infer.py` into side-arrays
   (`max_per_seq`, `argmax_per_seq`, `fire_count`, `decoder_directions`).
3. **Activations are stored raw**; the global normalisation scalar lives in
   `meta.json` (`extract.py`) and is folded into every SAE checkpoint
   (`train.py`). `infer.py` re-applies it. This keeps the SAE compatible
   with un-normalised activations at inference time.

## Restartability

Every stage no-ops if its primary output exists. To force a rerun of one
stage without nuking later stages, delete that stage's `meta.json` (plus
`features.npy` for `infer.py`).

## Testing

There are no unit tests. The standard sanity loop is:

```bash
# Smoke-test imports
.venv/bin/python -c "import download, extract, train, infer, analyse"

# Lint
.venv/bin/python -m pyflakes *.py

# Small end-to-end run
MONO_EXPANSION_FACTOR=4 \
MONO_NUM_EXTRACT_TOKENS_TRAIN=20480 \
MONO_NUM_EXTRACT_TOKENS_TEST=10240 \
MONO_NUM_TRAINING_STEPS=200 \
MONO_CHECKPOINT_EVERY=200 \
MONO_BATCH_SIZE=512 \
MONO_BUFFER_SEQUENCES=8 \
MONO_INFER_FEATURE_BLOCK=64 \
MONO_ANALYSIS_NUM_FEATURES=8 \
MONO_ANALYSIS_TOP_K=5 \
MONO_ANALYSIS_MIN_DISTINCT_SEQUENCES=3 \
  bash -c '.venv/bin/python download.py && \
           .venv/bin/python extract.py && \
           .venv/bin/python train.py && \
           .venv/bin/python infer.py && \
           .venv/bin/python analyse.py'
```

When testing UI changes, open `data/analysis/report.html` in a browser and
check the metadata table renders and at least one feature has visible
green-shaded tokens.
