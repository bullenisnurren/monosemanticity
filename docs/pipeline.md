# Pipeline walkthrough

Per-script reference: what each stage does, what it reads and writes, and
how to run it.

---

## 1. `download.py`

Snapshot the target model from HuggingFace and stream-download dataset shards.

**Inputs**: network. A valid HF token in `~/.cache/huggingface/token` is
required if the model is gated (Llama-3.2-1B is).

**Outputs**:
- `data/models/<model_slug>/` — full HF snapshot.
- `data/datasets/<dataset_slug>/data/shard_*.jsonl` — 50 000 rows each.
- `data/datasets/<dataset_slug>/meta.json` — counts.

**Skips if**: at least one `*.safetensors`/`*.bin` is present in the model
dir and `<dataset_slug>/meta.json` exists.

Run: `python download.py`.

---

## 2. `extract.py`

Forward-pass dataset shards through one transformer layer and capture
residual-stream activations. Produces a `train/` and a `test/` subdirectory
from a single sweep over the data (disjoint sequences).

**Inputs**: outputs of `download.py`.

**Outputs (per split)** under `data/activations/<model>/layer<N>/<split>/`:
- `activations.npy` — `(n_seqs, stored_seq_len, d_model)` fp32, raw
  (unnormalised). Each sequence's content comes from a single document, with
  `SEQ_LEN - n_prefix_tokens` content positions stored; the tokenizer's
  special-token prefix (e.g. BOS for Llama-3.2) is fed to the model for
  context but stripped from disk.
- `token_ids.npy` — `(n_seqs, stored_seq_len)` int32, content tokens only.
- `sequences.jsonl` — one `{"text": ..., "tokens": [...]}` per sequence
  (content only).
- `meta.json` — split metadata, including the global normalisation scalar
  `scale` (computed from the train split), the stored `seq_len`, the
  `model_seq_len` (the input length used for the forward pass) and
  `n_prefix_tokens`.

**Skips if**: both `train/meta.json` and `test/meta.json` exist.

Run: `python extract.py`.

---

## 3. `train.py`

Train the SAE on the train-split activations.

**Inputs**: `data/activations/<model>/layer<N>/train/`.

**Outputs**: `data/sae_checkpoints/sae_step_<step>.pt` every
`CHECKPOINT_EVERY` steps. Each checkpoint contains SAE state, optimiser /
scheduler state, the activation scale, and the training config.

Run: `python train.py`. If `data/sae_checkpoints/` already contains a
checkpoint, training **resumes** from it (model + optimiser + scheduler
state are all reloaded); if the latest checkpoint is already at
`NUM_TRAINING_STEPS`, the script exits immediately. To force a fresh run,
move or clear the checkpoint directory.

---

## 4. `infer.py`

Run the latest SAE checkpoint on the test-split activations and persist the
full feature tensor plus precomputed summaries that `analyse.py` needs.

**Inputs**: latest `data/sae_checkpoints/sae_step_*.pt` and
`data/activations/<model>/layer<N>/test/`.

**Outputs** under `data/features/<model>/layer<N>/`:
- `features.npy` — `(F, N, S)` fp16, feature-major.
- `fire_count.npy` — `(F,)` int64, firing count per feature.
- `max_per_seq.npy` — `(N, F)` fp16, peak activation per (sequence, feature).
- `argmax_per_seq.npy` — `(N, F)` int16, peak-token index within sequence.
- `decoder_directions.npy` — `(F, d)` fp16, unit-norm decoder rows.
- `token_ids.npy`, `sequences.jsonl` — copied from the test activations dir.
- `meta.json` — checkpoint info, layout descriptor, file pointers.

**Skips if**: `meta.json` and `features.npy` both exist in
`data/features/<model>/layer<N>/`.

Run: `python infer.py`.

---

## 5. `analyse.py`

Build the HTML report. No GPU work; never invokes the SAE or the studied LM.

**Inputs**: `data/features/<model>/layer<N>/`.

**Output**: `data/analysis/report.html`.

What it does:
1. Loads metadata + side arrays.
2. Filters candidate features by firing rate (Goldilocks band) and minimum
   number of distinct activating sequences.
3. Samples features — either by greedy farthest-point on decoder cosine
   (default) or uniformly at random over the candidates.
4. For each sampled feature, finds the top-k sequences via
   `max_per_seq` / `argmax_per_seq` and reads per-token activations via a
   single contiguous slice of `features.npy`.
5. POSTs to an OpenAI-compatible `/v1/chat/completions` endpoint for one
   natural-language description per feature.
6. Renders an HTML report with token-level green highlighting proportional
   to per-token activation.

Run: `python analyse.py`. Cheap to rerun (no GPU, ~seconds of disk I/O plus
the LLM calls).
