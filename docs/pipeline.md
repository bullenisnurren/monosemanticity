# Pipeline walkthrough

Per-script breakdown — what each stage does, where it writes, and how to test
it in isolation. For the higher-level rationale, see
[architecture.md](architecture.md); for the on-disk file layouts, see
[data-formats.md](data-formats.md).

---

## 1. `download.py`

Snapshot the model and stream-download the dataset.

**Inputs:** internet, valid HF token in `~/.cache/huggingface/token` if the
model is gated (Llama-3.2-1B is).

**Outputs:**
- `data/models/<model_slug>/` — full HF snapshot (config, tokenizer,
  `model.safetensors`).
- `data/datasets/<dataset_slug>/data/shard_*.jsonl` — JSONL shards of dataset
  rows (50 000 rows per shard).
- `data/datasets/<dataset_slug>/meta.json` — number of examples, shards, etc.

**Skips if:** at least one `*.safetensors`/`*.bin` is present in the model
dir, and `<dataset_slug>/meta.json` exists.

**Quirks:**
- Dataset is downloaded *streaming*, so we only ever fetch the rows we'll
  use. The `_estimate_examples_needed` helper uses
  `target_tokens / seq_len × 1.5` as a heuristic example count.
- `monology/pile-uncopyrighted` ships as zstandard-compressed files;
  `requirements.txt` pins `zstandard` for this.

**Test it isolated:** just run `python download.py`. Idempotent.

---

## 2. `extract.py`

Forward-pass dataset shards through one transformer layer, capture
residual-stream activations.

**Inputs:** model + dataset from previous stage.

**Outputs (per split):**
- `data/activations/<model>/layer<N>/<split>/activations.npy` —
  shape `(n_seqs, SEQ_LEN, d_model)`, fp32, *raw* (unnormalised).
- `token_ids.npy` — shape `(n_seqs, SEQ_LEN)`, int32.
- `sequences.jsonl` — one JSON per sequence: `{"text": ..., "tokens": [...]}`
  (the per-token strings come from `tokenizer.decode([id])` per token, so
  analyse.py can render them without loading the tokenizer).
- `meta.json` — `{model_name, layer_index, d_model, seq_len, scale,
  num_sequences, num_tokens, split, …}`.

**Skips if:** both `train/meta.json` and `test/meta.json` exist.

**Implementation notes:**
- Model loaded in fp16. With `NUM_GPUS=1` it's placed on cuda:0; with
  `NUM_GPUS≥2` we let `accelerate`'s `device_map="auto"` shard the LM and
  fence other GPUs out via a `max_memory` dict (so e.g. a busy cuda:1 is
  ignored).
- A single sequence iterator (`_iter_sequences`) is constructed once and
  shared between the two `_extract_split` calls — train pulls its sequences
  first, test pulls *the very next ones* (no overlap, no leak).
- Activations are streamed to a raw binary file, then a `.npy` header is
  prepended with the final 3D shape. Avoids holding the entire tensor in RAM.
- The normalisation scalar
  `scale = √(d_model / E[‖x‖²])` is computed using `sum_sq_norms` accumulated
  during the *train* pass. The test split's `meta.json` records the same
  scale, so anything loading either split is consistent.

**Test it isolated:**

```bash
MONO_NUM_GPUS=1 \
MONO_NUM_EXTRACT_TOKENS_TRAIN=20480 \
MONO_NUM_EXTRACT_TOKENS_TEST=10240 \
  python extract.py
```

Should finish in well under a minute and produce a few MB under
`data/activations/...`.

---

## 3. `train.py`

Train the SAE on the train-split activations.

**Inputs:** `data/activations/<model>/layer<N>/train/`.

**Outputs:** `data/sae_checkpoints/sae_step_<N>.pt` — one per `CHECKPOINT_EVERY`
steps. Each is a dict containing the SAE state dict, optimiser/scheduler
state, the activation scale (so infer.py can apply it), and a `config` dict.

**Implementation notes:**

### The SAE (`SparseAutoencoder`)
- Decoder `W_dec` shape `(d, F)`; encoder `W_enc` shape `(F, d)`.
  Both initialised with `W_dec` ~ random direction × `DECODER_INIT_NORM`,
  then `W_enc = W_dec.T`.
- `encode(x)`: ReLU(W_enc · (x − b_dec) + b_enc).
- `decode(f)`: W_dec · f + b_dec.
- `forward(x, l1)` returns `(loss, mse_loss, l1_loss, l0)` where
  `l1_loss = Σᵢ fᵢ · ‖w_dec_i‖` is the decoder-norm-weighted L1 and
  `l0 = mean(f > 0)` is the active-feature count per token.

### The activation loader (`ActivationLoader`)
Spinning-disk-aware:
- On-disk activations are `(n_seqs, S, d)` fp32, sequence-major C-order.
- A worker thread (`ThreadPoolExecutor` with one thread) repeatedly reads
  `BUFFER_SEQUENCES` random sequences into a fp32 RAM buffer, then shuffles
  *tokens* inside the buffer in place. Sorting the sequence IDs before
  reading ensures the disk head moves monotonically through the file.
- Normalisation (`× scale`) is folded into the buffer build, so
  `get_batch()` is a single slice + `torch.from_numpy(...).to(device)`.
- Quality argument: with `BUFFER_SEQUENCES ≫ batch_size`, each batch
  averages `batch_size / buffer_sequences ≪ 1` token from any one document,
  so within-batch correlation is negligible. Empirically indistinguishable
  from full token-level shuffling for SAE training.

### Training loop
- Adam (no weight decay).
- L1 coefficient linearly warmed up over the first `L1_WARMUP_FRAC` (5%) of
  steps.
- LR linearly decayed to 0 over the last `LR_DECAY_FRAC` (20%) of steps.
- Gradient norm clipped to `GRAD_CLIP_NORM = 1.0`.
- Multi-GPU: `nn.DataParallel`, which splits the input batch across GPUs
  (with `NUM_GPUS > 1`).

**Test it isolated:**

```bash
MONO_NUM_GPUS=1 \
MONO_EXPANSION_FACTOR=4 MONO_BATCH_SIZE=256 \
MONO_NUM_TRAINING_STEPS=50 MONO_CHECKPOINT_EVERY=50 \
MONO_BUFFER_SEQUENCES=8 \
  python train.py
```

---

## 4. `infer.py`

Run the latest SAE on the test split, persist features + side arrays for
`analyse.py`.

**Inputs:** latest `data/sae_checkpoints/sae_step_*.pt`,
`data/activations/<model>/layer<N>/test/`.

**Outputs (under `data/features/<model>/layer<N>/`):**
- `features.npy` — shape `(F, N, S)` fp16. **Feature-major** layout —
  rationale in [architecture.md](architecture.md).
- `fire_count.npy` — `(F,)` int64. Number of test tokens each feature fires on.
- `max_per_seq.npy` — `(N, F)` fp16. Peak activation per (sequence, feature).
- `argmax_per_seq.npy` — `(N, F)` int16. Peak-token index within sequence.
- `decoder_directions.npy` — `(F, d)` fp16. Unit-norm decoder rows
  (transposed so per-feature reads are contiguous).
- `token_ids.npy`, `sequences.jsonl` — copied verbatim from the test
  activations dir so analyse.py only needs one input directory.
- `meta.json` — checkpoint info, layout descriptor, file pointers.

**Skips if:** `meta.json` and `features.npy` both exist.

**Implementation notes:**
- All test activations (4 GB bf16) are loaded onto the GPU once; we
  then iterate **feature blocks** (`INFER_FEATURE_BLOCK = 512`).
- `b_dec` is folded into a "biased encoder bias"
  `b_enc_eff = b_enc − b_dec · W_enc.T` (precomputed in fp32) so we never
  need to allocate a centred copy of `x`.
- bf16 matmul on Ampere; features stored as fp16. SAE is cast to bf16
  before the block sweep.
- Per block:
  1. Compute `f_3d = ReLU(x · W_block.T + b_eff_block) · dec_norm_block`
     of shape `(N, S, B)` bf16.
  2. Accumulate stats: `fire_count`, `max_per_seq`, `argmax_per_seq`.
  3. Permute → fp16 → contiguous → numpy → write
     `features[f_start:f_end] = f_out`. This is the dominant cost.

**Performance:** at default sizes the bottleneck is HDD sequential write at
~280 MB/s, giving a ~22-minute floor for the 262 GB feature tensor. See
[performance.md](performance.md) for the budget analysis.

**Test it isolated:** can only run after `extract.py` + at least one
checkpoint exists. Drop the same scale-down env vars used for training to
verify a small end-to-end run.

---

## 5. `analyse.py`

Build the HTML report. No GPU work. Does *not* call the SAE or the studied LM.

**Inputs:** `data/features/<model>/layer<N>/`.

**Output:** `data/analysis/report.html`.

**Stages within `analyse()`:**

1. **Load metadata + side arrays.** `max_per_seq`, `argmax_per_seq`,
   `fire_count` (~1 GB total) are loaded fully into RAM.
   `features.npy` is mmap-ed.
2. **Filter candidates.**
   - Goldilocks band: keep features with `fire_count / total_tokens` ∈
     `[ANALYSIS_MIN_FIRE_FRAC, ANALYSIS_MAX_FIRE_FRAC]`.
   - Support: keep features with at least `ANALYSIS_MIN_DISTINCT_SEQUENCES`
     sequences where peak activation is positive.
3. **Sample features.**
   - If `ANALYSIS_DIVERSE_SELECTION` is on, use greedy farthest-point
     selection on decoder cosine similarity (loads
     `decoder_directions.npy` for the candidates).
   - Otherwise uniform random over the filtered candidates.
4. **Top-k per feature.** For each sampled feature `fid`:
   - `max_per_seq[:, fid]` gives per-sequence peaks → argpartition for top-k.
   - `features[fid]` is one 2 MB contiguous read; extract the per-token rows
     for the top-k seqs.
5. **LLM judge.** For each feature, build a prompt with up to
   `LLM_NUM_EXAMPLES` excerpts (centred on each peak token, truncated to
   `LLM_EXAMPLE_CHARS`) and POST to `LLM_API_BASE_URL/v1/chat/completions`
   via stdlib `urllib`. Failures are caught and surface as "(LLM call failed)".
6. **Render HTML.** Metadata tables (experiment + SAE config) followed by one
   block per sampled feature: description, stats, and the top-k token panels
   with `<span>`s coloured `rgba(46, 204, 113, intensity)` where intensity =
   per-token activation / per-feature peak.

**Tunable in `constants.py` (all `MONO_*`):**
- Selection — see `ANALYSIS_*`.
- LLM — see `LLM_*`.

**Test it isolated:** run after `infer.py`. Cheap (~30 s).
