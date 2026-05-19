# AGENTS.md — guidance for AI coding agents

This file is for AI assistants helping develop or modify this repository.
Humans should also find it useful as a "what to know before touching anything"
primer.

## What this repo is

A 5-stage pipeline implementing the
[Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity)
sparse-autoencoder (SAE) recipe on Llama-3.2-1B. The final artefact is an HTML
report describing what each SAE feature appears to detect, with token-level
heatmaps.

Stages, in order:
1. `download.py` — model + dataset to disk.
2. `extract.py` — residual-stream activations from one LM layer, split into train + test.
3. `train.py` — train the SAE on train activations.
4. `infer.py` — run the SAE on test activations, save the feature tensor + side-arrays.
5. `analyse.py` — render the HTML report (CPU only; calls one LLM API for descriptions).

Read `README.md` first for the user-facing summary, then `docs/architecture.md`
for the data-flow diagram, then `docs/pipeline.md` for per-script details.

## Project conventions

### Configuration → `constants.py` (single source of truth)

Every tunable is a module-level constant in `constants.py`, declared once with
a type annotation and a `_env("MONO_<NAME>", default, dtype)` call so it can be
overridden via env var. Example:

```python
ANALYSIS_TOP_K: int = _env("MONO_ANALYSIS_TOP_K", 20, int)
```

**When adding a new tunable**: put it in `constants.py`, follow the existing
naming (`MONO_<UPPER_SNAKE>`), and import it from the script that needs it.
Don't sprinkle magic numbers around.

### Path conventions

`constants.py` derives every path from `DATA_DIR = PROJECT_ROOT / "data"`. In
the test setup, `data` is a symlink to a fast/large external disk
(`/mnt/sda/...`). Anything written under `data/` is gitignored.

Per-model and per-layer artefacts are namespaced:
- `data/activations/<model_slug>/layer<N>/{train,test}/`
- `data/features/<model_slug>/layer<N>/`

`<model_slug>` is `MODEL_NAME` with `/` → `__`.

### Style

- Python 3.10+ syntax (`list[int]`, `dict[str, X]`, `str | None`).
- Type-annotated function signatures.
- Docstrings on every non-trivial function/class — explain *why*, not what.
- Comments only where the *why* is non-obvious; named identifiers carry the
  *what*.
- No emojis (unless a user explicitly asks).

### Dependencies

Listed in `requirements.txt`, pinned. Standard ML + numpy + tqdm. The LLM call
in `analyse.py` uses `urllib.request` (stdlib only) — do not introduce an
`openai`/`httpx` dep for that.

## Architecture invariants (don't break these)

1. **HDD-aware I/O.** Training activations are 60+ GB and live on a spinning
   disk. Per-batch random-token access (~4096 small reads scattered across
   60 GB) effectively never completes. The buffered loader in `train.py`
   exists for this reason — see `docs/performance.md`.

2. **Feature tensor is feature-major `(F, N, S)`.** This is what makes
   `analyse.py` fast: `features[fid]` is a contiguous 2 MB read.
   `infer.py` is structured around iterating over feature blocks so writes
   are also contiguous. Do *not* switch to `(N, S, F)` C-order, even though
   it might feel more natural — analyse.py becomes ~200× slower.

3. **`analyse.py` does no GPU work and never invokes the LM or the SAE.**
   It is supposed to be runnable on a laptop given the features directory.
   Everything analyse.py needs (per-(seq, feat) max + argmax, decoder
   directions, etc.) is precomputed by `infer.py` into side-arrays.

4. **Raw activations + normalisation scalar.** `extract.py` saves
   *unnormalised* activations and a single global `scale` in `meta.json`.
   `train.py` applies the scale at batch load time and stores it in each
   checkpoint. `infer.py` reads the scale from the checkpoint so it can be
   re-applied to raw activations. This keeps the SAE compatible with raw
   activations at inference time.

5. **Decoder column norms scale the features.** The L1 penalty is
   `Σᵢ fᵢ · ‖w_dec_i‖` during training (paper §4); after training, when we
   present features for analysis we use the "true unit" formulation
   `f_i' = f_i · ‖w_dec_i‖` so the values are intrinsically comparable
   across features. `infer.py` performs this scaling once.

## Common gotchas

- **`MONO_NUM_GPUS=1`** is required on the test machine because cuda:1/cuda:2
  are usually busy. Forgetting this OOMs on a wrong GPU. CPU-only paths are
  not supported.
- **GPU memory in `infer.py`** is tight (~13 GB peak). The `b_dec` fold trick
  (`b_enc_eff = b_enc − b_dec @ W_enc.T`) and bf16 inference are load-bearing
  for fitting in 24 GB.
- **Memmap RSS in `top`** can show ~90% during `infer.py`. This is the Linux
  page cache backing dirty mmap pages, not a leak. The kernel caps it via
  `vm.dirty_ratio` and reclaims under pressure. See `docs/performance.md`.
- **Restart-friendliness**: every stage no-ops if its primary output exists.
  Deleting `meta.json` is the supported way to force a rerun without nuking
  the bulk data first.
- **`features.npy` is huge** (~245 GB on default settings). The disk write
  during `infer.py` is the wall-clock floor — there's nothing to optimise
  beyond the HDD's sequential write speed. See `docs/performance.md` for
  the budget analysis.

## How to test / iterate

There are no unit tests yet. The standard sanity loop is:

```bash
# Smoke-test imports
.venv/bin/python -c "import download, extract, train, infer, analyse; print('OK')"

# Lint
.venv/bin/python -m pyflakes *.py

# End-to-end smoke at small scale (overrides keep it minute-ish):
MONO_NUM_GPUS=1 \
MONO_EXPANSION_FACTOR=4 \
MONO_NUM_EXTRACT_TOKENS_TRAIN=20480 \
MONO_NUM_EXTRACT_TOKENS_TEST=10240 \
MONO_NUM_TRAINING_STEPS=50 \
MONO_CHECKPOINT_EVERY=50 \
MONO_BATCH_SIZE=256 \
MONO_BUFFER_SEQUENCES=8 \
MONO_INFER_FEATURE_BLOCK=64 \
MONO_ANALYSIS_NUM_FEATURES=4 \
MONO_ANALYSIS_TOP_K=3 \
MONO_ANALYSIS_MIN_DISTINCT_SEQUENCES=2 \
  .venv/bin/python download.py && \
  .venv/bin/python extract.py && \
  .venv/bin/python train.py && \
  .venv/bin/python infer.py && \
  .venv/bin/python analyse.py
```

When testing UI-ish changes (report rendering), open
`data/analysis/report.html` in a browser. Always verify both:
1. The metadata table renders correctly.
2. At least one feature has visible green-shaded tokens proportional to
   activations.

## Performance budgets (single 3090 + spinning disk)

| Stage         | Wall clock     | Bottleneck                                    |
|---------------|----------------|-----------------------------------------------|
| `download.py` | ~6 min         | model download (one-time)                     |
| `extract.py`  | ~8 min         | LM forward + tokenizer.decode CPU             |
| `train.py`    | ~22 min / 50K steps | SAE forward/backward on GPU              |
| `infer.py`    | ~22 min        | HDD sequential write floor (262 GB / 200 MB/s) |
| `analyse.py`  | ~30 s          | LLM call latency (network + inference)        |

Targeting changes against these numbers helps avoid accidental regressions.

## Things that are *not* worth doing

These have been considered and ruled out — don't bring them back without
strong reason.

- **Storing features in `(N, S, F)` C-order.** Convenient, but kills
  analyse.py.
- **Computing all 100 LLM descriptions in parallel.** The local 26 B model
  saturates at ~3 req/s anyway; ThreadPoolExecutor adds complexity for
  little wall-clock gain at the default `ANALYSIS_NUM_FEATURES = 100`.
- **`posix_fadvise(POSIX_FADV_DONTNEED)` in infer.py.** Would require an
  `msync` to flush dirty pages first, which serialises writes and slows
  infer.py. The kernel's `vm.dirty_ratio` cap is sufficient.
- **HDF5 / Zarr for features.** Adds a heavy dependency for marginal benefit;
  per-feature row reads on the feature-major numpy `.npy` are already
  contiguous and fast.

## Where to look for things

- "How do I add a tunable?" → `constants.py`.
- "What's the on-disk format of X?" → `docs/data-formats.md`.
- "Why is the loader/storage layout the way it is?" → `docs/architecture.md` + `docs/performance.md`.
- "How do I run X with smaller settings to debug?" → `docs/pipeline.md` per-stage table.
- "What does this constant do?" → `docs/configuration.md`.
