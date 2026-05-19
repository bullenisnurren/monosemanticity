# Data formats

Authoritative reference for every file the pipeline writes. Shapes assume the
default constants (`d_model = 2048`, `SEQ_LEN = 512`, `EXPANSION_FACTOR = 64`,
`NUM_EXTRACT_TOKENS_TRAIN = 20M`, `NUM_EXTRACT_TOKENS_TEST = 1M`, so
`n_seqs_train ≈ 39 062`, `n_seqs_test = 1 953`, `dict_size = 131 072`).

---

## `data/models/<model_slug>/`

Plain HuggingFace snapshot, untouched. Contains:

- `config.json`, `tokenizer*.json`, `special_tokens_map.json`,
  `generation_config.json` — model config + tokenizer.
- `model.safetensors` — weights (~2.4 GB for Llama-3.2-1B).
- `LICENSE.txt`, `README.md`, `USE_POLICY.md`, `.gitattributes`, etc.
- `original/`, `.cache/huggingface/` — HF internals; ignore.

`<model_slug>` = `MONO_MODEL_NAME` with `/` replaced by `__`.

---

## `data/datasets/<dataset_slug>/`

- `data/shard_00000.jsonl`, `shard_00001.jsonl`, … — 50 000 rows each.
  Every row is the dict from the original HuggingFace dataset; we read the
  `MONO_DATASET_TEXT_FIELD` (`text` by default).
- `meta.json` — `{dataset_name, split, text_field, num_examples, num_shards}`.

`<dataset_slug>` = `MONO_DATASET_NAME` with `/` replaced by `__`.

---

## `data/activations/<model_slug>/layer<N>/{train,test}/`

One directory per split. Both have the same file layout — they differ only
in size.

### `activations.npy`
- Shape: `(num_sequences, SEQ_LEN, d_model)`.
- dtype: `float32`.
- Layout: C-major (innermost axis = `d_model`).
- Content: **raw** residual-stream activations at layer `LAYER_INDEX`,
  no normalisation applied. Apply `× scale` (from `meta.json`) at read
  time to get the values the SAE was trained on.

Size at defaults:
- Train: `39 062 × 512 × 2048 × 4 B ≈ 153 GB`.
- Test:  `1 953 × 512 × 2048 × 4 B ≈ 8.2 GB`.

### `token_ids.npy`
- Shape: `(num_sequences, SEQ_LEN)`.
- dtype: `int32`.
- Content: HF tokenizer IDs for each token in each sequence. Sequences are
  concatenated and chopped into `SEQ_LEN`-token windows (no padding,
  document boundaries are not preserved).

### `sequences.jsonl`
- One JSON per line per sequence:
  `{"text": "<decoded sequence>", "tokens": ["<tok0>", "<tok1>", …]}`
- `tokens[i] = tokenizer.decode([token_ids[i]])` — per-token strings,
  precomputed in extract.py so analyse.py doesn't need a tokenizer.

### `meta.json`

```jsonc
{
  "model_name": "meta-llama/Llama-3.2-1B",
  "layer_index": 8,
  "d_model": 2048,
  "seq_len": 512,
  "scale": 1.2919716610690406,    // √(d_model / E[‖x‖²]) computed on train
  "act_file": "activations.npy",
  "token_ids_file": "token_ids.npy",
  "sequences_file": "sequences.jsonl",
  "split": "train",                // or "test"
  "num_sequences": 39062,
  "num_tokens": 19999744
}
```

`scale` is identical in both splits (always computed from train).

---

## `data/sae_checkpoints/sae_step_<step>.pt`

`torch.save` of a dict:

```python
{
  "step": 52000,
  "d_model": 2048,
  "dict_size": 131072,
  "expansion_factor": 64,
  "activation_scale": 1.2919716610690406,
  "model_state_dict": {...},        # SAE weights (W_enc, W_dec, b_enc, b_dec)
  "optimiser_state_dict": {...},    # Adam state for resuming
  "scheduler_state_dict": {...},    # LambdaLR state
  "config": {                       # snapshot of training hyperparams
    "l1_coeff": 5.0,
    "lr": 5e-05,
    "batch_size": 4096,
    "num_training_steps": 200000,
    "decoder_init_norm": 0.1,
  },
}
```

Size at defaults: ~6 GB per checkpoint (SAE parameters + Adam state in fp32).

`activation_scale` is read by `infer.py` to apply the same normalisation that
was used during training.

---

## `data/features/<model_slug>/layer<N>/`

### `features.npy`
- Shape: `(dict_size, num_test_sequences, seq_len)`.
- dtype: `float16` (configurable via `MONO_FEATURE_DTYPE`).
- Layout: **C-major, feature-major** — `features[fid]` is contiguous on disk.
- Content: `f = ReLU(W_enc·(x·scale − b_dec) + b_enc) · ‖w_dec_i‖`. The
  trailing `‖w_dec_i‖` factor scales features into the "true-unit" formulation
  so values are intrinsically comparable across features.

Size at defaults: `131 072 × 1 953 × 512 × 2 B ≈ 262 GB`.

### `fire_count.npy`
- Shape: `(dict_size,)`.
- dtype: `int64`.
- Content: number of test tokens on which each feature fires (`f > 0`).
  Dead features have `0` here.

### `max_per_seq.npy`
- Shape: `(num_test_sequences, dict_size)`.
- dtype: `float16`.
- Content: `max_per_seq[n, f] = max_t features[f, n, t]`. Used by analyse.py
  to find the top-k sequences per feature without scanning `features.npy`.

### `argmax_per_seq.npy`
- Shape: `(num_test_sequences, dict_size)`.
- dtype: `int16`.
- Content: `argmax_per_seq[n, f] = argmax_t features[f, n, t]`. The token
  index *within* each sequence where the feature peaks. Cast to int16
  because `SEQ_LEN ≤ 32767` always.

### `decoder_directions.npy`
- Shape: `(dict_size, d_model)`.
- dtype: `float16`.
- Layout: C-major; `decoder_directions[fid]` is the unit-norm decoder
  direction for feature `fid`. (Transposed from the SAE's internal
  `W_dec` shape so per-feature reads are contiguous.)
- Used by analyse.py's diversity selection (greedy farthest-point on cosine).

### `token_ids.npy`, `sequences.jsonl`
Verbatim copies of the test-split equivalents. Living here means analyse.py
needs only one input directory.

### `meta.json`

```jsonc
{
  "checkpoint": "sae_step_052000.pt",
  "step": 52000,
  "activation_scale": 1.2919716610690406,
  "d_model": 2048,
  "dict_size": 131072,
  "expansion_factor": 64,
  "num_sequences": 1953,
  "seq_len": 512,
  "feature_dtype": "float16",
  "feature_decoder_scaled": true,           // features are pre-multiplied by ‖w_dec‖
  "features_layout": "feature_major",       // (F, N, S) C-order
  "features_shape": [131072, 1953, 512],
  "features_file": "features.npy",
  "fire_count_file": "fire_count.npy",
  "max_per_seq_file": "max_per_seq.npy",
  "argmax_per_seq_file": "argmax_per_seq.npy",
  "decoder_directions_file": "decoder_directions.npy",
  "token_ids_file": "token_ids.npy",
  "sequences_file": "sequences.jsonl",
  "test_split_meta": { ... },               // verbatim copy of test/meta.json
  "training_config": { ... }                // verbatim copy of ckpt["config"]
}
```

---

## `data/analysis/report.html`

Self-contained HTML — inlined CSS, no external assets. Structure:

1. `<h1>SAE Feature Report</h1>`.
2. `<h2>Experiment metadata</h2>` + table (model, layer, dataset, scale, etc.).
3. `<h2>SAE metadata</h2>` + table (checkpoint, dict size, alive/dead
   features, key training hyperparams).
4. `<h2>Sampled features</h2>` + one `<div class="feature">` per sampled
   feature:
   - `<h3>Feature #fid</h3>`
   - `<p class="desc">` — LLM-generated natural-language description.
   - `<p class="stats">` — top-example count, peak activation, firing rate.
   - One `<div class="example">` per top-k sequence. Each token wrapped in
     `<span class="tok" style="background-color: rgba(46, 204, 113, α)">`
     where `α ∈ [0, 1]` is the per-token activation divided by the feature's
     overall peak.

Size at defaults: ~25–30 MB (100 features × 20 examples × 512 tokens per
example).
