# Architecture

This document explains *why* the code looks the way it does. For *what each
script does step-by-step*, see [pipeline.md](pipeline.md).

## Data flow

```
+-------------+      +-------------+      +-------------+
| HuggingFace |      | HuggingFace |      |  Anthropic  |
|    model    |      |   dataset   |      |    paper    |
+------+------+      +------+------+      +-------------+
       |                    |
       v                    v
+-------------------------------------+
|         download.py                 |
|  ./data/models/<m>/                 |
|  ./data/datasets/<d>/data/*.jsonl   |
+-------------------+-----------------+
                    |
                    v
+-------------------------------------+
|         extract.py                  |
|  (forward through one LM layer)     |
|  ./data/activations/<m>/layer<N>/   |
|    train/  activations.npy  (3D)    |
|            token_ids.npy            |
|            sequences.jsonl          |
|            meta.json     (scale)    |
|    test/   activations.npy  (3D)    |
|            token_ids.npy            |
|            sequences.jsonl          |
|            meta.json                |
+-------------------+-----------------+
                    |
        +-----------+-----------+
        | train                 | test
        v                       v
+---------------------+    +-----------------------------+
|     train.py        |    |        infer.py             |
| (buffered loader,   |    | (feature-block sweep,        |
|  SAE training)      |    |  saves feature-major tensor) |
| ./data/sae_         |    | ./data/features/<m>/layer<N>/|
|    checkpoints/     |    |  features.npy   (F, N, S)    |
|   sae_step_*.pt     |    |  max_per_seq.npy  (N, F)     |
+----------+----------+    |  argmax_per_seq.npy (N, F)   |
           |               |  fire_count.npy   (F,)       |
           +-------------> |  decoder_directions.npy (F,d)|
                           |  meta.json                   |
                           +---------------+--------------+
                                           |
                                           v
                           +-------------------------------+
                           |        analyse.py             |
                           | (CPU only; calls LLM judge)   |
                           | ./data/analysis/report.html   |
                           +-------------------------------+
```

The five scripts are intentionally pipelineable: each writes a stable on-disk
contract, and each no-ops if its main output (typically a `meta.json`) is
already present. You can rerun `analyse.py` 50 times without re-running
anything upstream.

## Why these particular file formats / layouts?

The repo bumps into two constraints that drive most of the design:

1. **Spinning disk** for the bulk artefacts (a single ~2 TB SATA disk).
   Random small reads are catastrophic; sequential access is essential.
2. **24 GB GPU**. Comfortably enough for the SAE and the LM, but features
   for the entire test set definitely do not fit; we have to stream.

These two together explain almost everything about how the data is laid out.

### Activations (`extract.py` output)

Shape `(num_sequences, SEQ_LEN, d_model)`, fp32, C-major. Saved as a single
`.npy` per split (`train/`, `test/`). Three reasons:

- **3D, not flat** — sequences are the natural unit during extraction (one
  forward pass produces a contiguous `(B, S, d)` chunk), and downstream
  consumers usually want to walk the data sequence-by-sequence.
- **Raw / unnormalised** — `extract.py` computes the global normalisation
  scalar `s = √(d_model / E[‖x‖²])` *over the train split*, saves it in
  `meta.json`, and bakes it into each SAE checkpoint. Storing raw lets us
  evaluate the same SAE on un-normalised activations later (e.g. for
  intervention experiments) and keeps the file format independent of the
  training-time preprocessing choice.
- **C-major** — every sequence is a contiguous 4 MB blob on disk, so the
  buffered loader in `train.py` can do nice sequential reads at HDD speed.

### Features (`infer.py` output)

Shape `(dict_size, num_test_sequences, seq_len)`, fp16, C-major. This is the
load-bearing decision in the repo, so it warrants a paragraph.

The "obvious" layout `(N, S, F)` (sequence-major) is what you get if you
naively concatenate per-batch outputs from a forward sweep over sequences.
The problem is that `analyse.py` only ever looks at ~100 features out of
~131 072, but with C-major `(N, S, F)` those 100 features are scattered
across every page of every row — reading them requires pulling the *entire*
262 GB tensor through the page cache. We'd be I/O-bound twice (once on write,
once on read).

Flipping to `(F, N, S)` makes `features[fid]` a single 2-MB contiguous read,
so analyse.py reads ~200 MB total instead of 262 GB. The cost is paid at
write time: `infer.py` has to iterate over **feature blocks** (rather than
sequence blocks) so its writes are also contiguous. This requires slicing
`W_enc` per block and computing features for all sequences at once. Total
disk write is still 262 GB (same bytes either way) — the storage layout
just dictates *how* those bytes are laid out.

The side-arrays (`max_per_seq`, `argmax_per_seq`, `fire_count`,
`decoder_directions`) are computed in the same sweep and sit next to
`features.npy`. They let analyse.py find dead features, identify top-k
sequences per feature, and apply decoder-cosine diversity selection
without touching the big tensor at all.

### Checkpoints (`train.py` output)

Plain `torch.save` `.pt` files, each holding the state dict, optimiser /
scheduler state, the activation normalisation scalar, and a config dict.
Big (~6 GB at default `EXPANSION_FACTOR=64`) but standard.

## The SAE itself

`SparseAutoencoder` (in `train.py`) follows the paper recipe:

- `f = ReLU( W_enc · (x − b_dec) + b_enc )` — encode (with the data-mean
  bias subtracted so the encoder sees zero-centred input).
- `x̂ = W_dec · f + b_dec` — decode.
- Loss: `MSE(x, x̂) + λ · Σᵢ fᵢ · ‖w_dec_i‖` — reconstruction + a *weighted*
  L1 where each feature's L1 contribution is multiplied by its decoder
  column norm. This is equivalent to L1 on the "true-unit" features
  `f_i' = f_i · ‖w_dec_i‖` and removes the need for explicit unit-norm
  decoder constraints.
- Init: decoder columns sampled from `N(0, I)` then scaled to a fixed norm
  (`DECODER_INIT_NORM = 0.1`); encoder initialised as `W_dec.T`.

At training time, decoder columns are not constrained to unit norm. After
training (in `infer.py`), we pre-compute `‖w_dec_i‖` once and scale features
by it before storing them — so feature magnitudes in the final tensor are
the "true unit" values, directly comparable across features.

## Why `analyse.py` is GPU-free

All of the operations that need the LM or the SAE itself are pushed into
`infer.py`. By the time `analyse.py` runs, everything it needs is in
`data/features/<model>/layer<N>/`:

- Per-token activations for any feature → slice `features[fid]`.
- Per-(seq, feat) peak + argmax for fast top-k → `max_per_seq`, `argmax_per_seq`.
- Decoder directions for diversity selection → `decoder_directions`.
- Sequences (token IDs and decoded text) → copied from the test-activations dir.

This lets `analyse.py` run on a laptop given the features directory, and
makes it cheap to re-render the report repeatedly with different selection
parameters or with a different LLM-judge.

## Pipeline restartability

Every stage has a simple "skip if output exists" guard:

- `download.py` checks for `*.safetensors`/`*.bin` weights and `meta.json` in
  the dataset dir.
- `extract.py` checks for both `train/meta.json` *and* `test/meta.json`.
- `train.py` doesn't auto-skip (training is fundamentally a "until step N"
  operation), but resumes naturally from the latest checkpoint via the
  optimiser/scheduler state. (Implementing literal "skip if done" is the
  one TODO.)
- `infer.py` checks for both `meta.json` and `features.npy` in the features
  dir.
- `analyse.py` always re-renders the report (cheap).

To force a rerun of one stage without nuking outputs of later stages,
delete that stage's `meta.json` (and `features.npy` for `infer.py`).
