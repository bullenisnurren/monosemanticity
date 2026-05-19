# Data formats

On-disk file specs for every artefact the pipeline writes. Shape examples
use the default constants (`d_model = 2048`, `SEQ_LEN = 512`,
`EXPANSION_FACTOR = 64`, so `dict_size = 131 072`).

---

## `data/models/<model_slug>/`

Untouched HuggingFace snapshot: `config.json`, `tokenizer*.json`,
`model.safetensors`, etc. `<model_slug>` = `MONO_MODEL_NAME` with `/` → `__`.

---

## `data/datasets/<dataset_slug>/`

- `data/shard_*.jsonl` — 50 000 rows each; each line is the original HF
  dataset row.
- `meta.json` — `{dataset_name, split, text_field, num_examples, num_shards}`.

---

## `data/activations/<model_slug>/layer<N>/{train,test}/`

### `activations.npy`
Shape `(num_sequences, SEQ_LEN, d_model)`, fp32, C-major. **Raw** residual
activations (no normalisation applied). Apply `× scale` (from `meta.json`)
to get the values the SAE expects.

### `token_ids.npy`
Shape `(num_sequences, SEQ_LEN)`, int32. HF tokenizer IDs.

### `sequences.jsonl`
One JSON per sequence: `{"text": "<decoded>", "tokens": ["<tok0>", ...]}`.
Per-token strings are precomputed so analyse.py doesn't need a tokenizer.

### `meta.json`

```jsonc
{
  "model_name": "meta-llama/Llama-3.2-1B",
  "layer_index": 8,
  "d_model": 2048,
  "seq_len": 512,
  "scale": 1.291...,                // sqrt(d_model / E[‖x‖²]) computed on train
  "act_file": "activations.npy",
  "token_ids_file": "token_ids.npy",
  "sequences_file": "sequences.jsonl",
  "split": "train",                 // or "test"
  "num_sequences": 39062,
  "num_tokens": 19999744
}
```

`scale` is identical in both splits.

---

## `data/sae_checkpoints/sae_step_<step>.pt`

`torch.save` of a dict:

```python
{
  "step": 52000,
  "d_model": 2048,
  "dict_size": 131072,
  "expansion_factor": 64,
  "activation_scale": 1.291...,
  "model_state_dict": {...},          # SAE weights
  "optimiser_state_dict": {...},
  "scheduler_state_dict": {...},
  "config": {                         # snapshot of training hyperparams
    "l1_coeff": 5.0, "lr": 5e-05,
    "batch_size": 4096, "num_training_steps": 200000,
    "decoder_init_norm": 0.1,
  },
}
```

---

## `data/features/<model_slug>/layer<N>/`

### `features.npy`
Shape `(dict_size, num_test_sequences, seq_len)`, fp16, C-major
(feature-major layout). Values are decoder-norm-scaled: `f_i · ‖w_dec_i‖`.

### `fire_count.npy`
Shape `(dict_size,)`, int64. Number of test tokens each feature fires on.

### `max_per_seq.npy`
Shape `(num_test_sequences, dict_size)`, fp16. Peak activation per
(sequence, feature).

### `argmax_per_seq.npy`
Shape `(num_test_sequences, dict_size)`, int16. Peak-token index within
sequence.

### `decoder_directions.npy`
Shape `(dict_size, d_model)`, fp16. Unit-norm decoder rows (transposed from
the SAE's internal `W_dec` so per-feature reads are contiguous).

### `token_ids.npy`, `sequences.jsonl`
Verbatim copies of the test-split equivalents.

### `meta.json`

```jsonc
{
  "checkpoint": "sae_step_052000.pt",
  "step": 52000,
  "activation_scale": 1.291...,
  "d_model": 2048,
  "dict_size": 131072,
  "expansion_factor": 64,
  "num_sequences": 1953,
  "seq_len": 512,
  "feature_dtype": "float16",
  "feature_decoder_scaled": true,
  "features_layout": "feature_major",
  "features_shape": [131072, 1953, 512],
  "features_file": "features.npy",
  "fire_count_file": "fire_count.npy",
  "max_per_seq_file": "max_per_seq.npy",
  "argmax_per_seq_file": "argmax_per_seq.npy",
  "decoder_directions_file": "decoder_directions.npy",
  "token_ids_file": "token_ids.npy",
  "sequences_file": "sequences.jsonl",
  "test_split_meta": { ... },         // verbatim copy of test/meta.json
  "training_config": { ... }          // verbatim copy of ckpt["config"]
}
```

---

## `data/analysis/report.html`

Self-contained HTML (inlined CSS, no external assets). Sections:

1. Experiment metadata table (model, layer, dataset, scale, etc.).
2. SAE metadata table (checkpoint, dict size, alive/dead features, training
   config).
3. One block per sampled feature: LLM-generated description, peak-activation
   stats, and the top-k sequences rendered as `<span>`s with
   `rgba(46, 204, 113, α)` backgrounds where `α` is the per-token activation
   divided by the feature's overall peak.
