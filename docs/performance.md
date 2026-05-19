# Performance & bottleneck analysis

Every stage in this pipeline has a clearly identifiable bottleneck. Knowing
which lets you avoid micro-optimisations that don't matter and target the
ones that do.

Numbers below assume the reference setup: **single RTX 3090 (24 GB),
~120 GB RAM, a single SATA HDD at ~280 MB/s sequential**.

---

## Budget summary

| Stage         | Wall clock     | Dominant cost                                     | What to optimise          |
|---------------|----------------|---------------------------------------------------|---------------------------|
| `download.py` | ~6 min         | Model + dataset download                          | Nothing local             |
| `extract.py`  | ~8 min         | LM forward (`d_model = 2048`, fp16)               | Reduce `NUM_EXTRACT_TOKENS_*` or skip layers |
| `train.py`    | ~22 min / 50K steps | SAE forward+backward on GPU                  | Larger batch (if RAM allows) |
| `infer.py`    | ~22 min        | **HDD sequential write floor** (262 GB)           | Smaller `EXPANSION_FACTOR` or test set; nothing else helps |
| `analyse.py`  | ~30 s          | LLM API latency (~100 × ~0.3 s)                   | Parallel LLM calls (not currently implemented) |

---

## The HDD bottleneck (everything else flows from this)

For the default configuration, three artefacts live on the HDD:

1. **Train activations**: ~153 GB.
2. **Test activations**: ~8 GB.
3. **Features tensor**: ~245 GB.

Sequential read/write of a SATA HDD tops out around 200–300 MB/s. Anything
that translates into random small reads collapses to single-digit MB/s.
This single fact drove three of the more "exotic" parts of the design:

- **`train.py`'s buffered loader.** Naively shuffling activations at the
  token level triggers `BATCH_SIZE = 4096` random 8 KB reads per step
  scattered across the 153 GB activations file. Even with the OS page cache
  warming up, the working set is too big to fit and per-step time was
  measured at ~24 s. The buffered loader (read whole sequences in sorted
  order into RAM, shuffle there, drain) brings this to ~10 ms/step — the
  GPU becomes the bottleneck again.

- **`infer.py`'s feature-major layout.** Writing 262 GB to HDD always takes
  ~25 min sequential. What we *don't* want is to then have to read those
  262 GB again in analyse.py to find top-k for 100 features. By storing
  features as `(F, N, S)` instead of `(N, S, F)`, per-feature reads are
  contiguous 2 MB chunks, and analyse.py reads ~200 MB total.

- **`extract.py`'s split sharing.** Both train and test share the same
  underlying sequence iterator, so we make a single sequential pass over the
  downloaded JSONL shards. No re-reading the dataset twice.

---

## `infer.py` deep dive

This is the longest single stage. Its time is essentially `262 GB / disk
write bandwidth`. There is no algorithmic win to be had — only smaller
inputs.

Observed behaviour (from `time -v`):
```
Elapsed (wall clock):   22:48
File system outputs:    513 970 552 (512-byte blocks) = 263 GB
```

`262 GB / 1368 s ≈ 196 MB/s` average. The actual disk speed is ~280 MB/s
sequential; the gap comes from the early blocks landing in the OS page
cache (≤ `vm.dirty_background_ratio`) before any disk I/O happens at all,
and from the kernel's `vm.dirty_ratio` cap throttling writes once cache
fills.

**Why the GPU is idle most of the time**: per-block compute is ~50 ms
(matmul `(N·S, d) × (d, B)` for `B = 512` at default `d = 2048` and
`N·S = 1 M`, in bf16 on a 3090). Per-block disk write is ~5 s once we're
in steady state. So GPU utilisation during infer.py is ~1%. This is fine;
it's the correct tradeoff for not OOMing the GPU on a 24 GB card.

**Memory pattern**: process RSS climbs to several tens of GB during the
run. This is the kernel page cache backing the memmap, not a Python leak.
- Dirty pages cap at `vm.dirty_ratio × MemTotal` (~25 GB on this machine).
- Clean cache pages of the written file fill whatever cache space remains,
  but are immediately reclaimable. `posix_fadvise(POSIX_FADV_DONTNEED)`
  would force-evict them, but only after an `msync`, which would serialise
  writes and make the run slower; the kernel defaults are optimal here.

---

## `train.py` deep dive

After the buffered loader fix, the loop is GPU-bound:

- Per-step compute: SAE forward (`x → f`, dict_size × d MAC) + reconstruction
  + backward + Adam. On a 3090 in fp32 this is ~20 ms per step.
- Data loader: ~5 GB buffer; refilled on a background thread. Refill takes
  ~30 s sequential (HDD-bound), drained over ~200 batches × 20 ms = 4 s of
  GPU time. So the GPU stalls about 26 s per refill — that's the headline
  inefficiency, but doubling `BUFFER_SEQUENCES` doesn't help because we're
  already disk-bound on the refill itself.

To make `train.py` faster on this hardware:
- **Store activations on SSD.** Most direct win. Activations are 153 GB,
  fits on a 200 GB SSD comfortably. Refill rate would jump to ~3 GB/s and
  the GPU would never stall.
- **Bigger batch.** GPU utilisation goes up; per-token compute time goes
  down. Capped by GPU memory.
- **Use bf16 in training.** Currently the SAE trains in fp32 (paper recipe).
  Switching to bf16/mixed precision could ~2× throughput but requires
  validation that final feature quality is unaffected.

---

## `analyse.py` deep dive

Operations and approximate times:

| Step                           | Cost              | Notes                                                              |
|--------------------------------|-------------------|--------------------------------------------------------------------|
| Load metadata + side arrays    | < 1 s             | ~1 GB total (`max_per_seq`, `argmax_per_seq`, `fire_count`).      |
| Filter candidates              | < 1 s             | Two vectorised numpy ops over (N, F).                              |
| Diverse selection              | ~1 s              | Greedy farthest-point on (~50K, d=2048) decoder rows.              |
| Top-k discovery                | ~1 s              | 100 × 2 MB contiguous reads from `features.npy`.                   |
| LLM descriptions               | ~30 s             | 100 sequential POSTs to a local LLM at ~3 req/s.                   |
| HTML render                    | < 1 s             | String building, no I/O.                                           |

Optimisation opportunities:
- **Parallel LLM calls** via `ThreadPoolExecutor` would cut the dominant cost
  by ~5× for `NUM_FEATURES = 100`. Not implemented because (a) 30 s is fine
  and (b) some local LLM servers throttle aggressively under concurrent load.

---

## Tuning recipes

### "I don't have enough disk space"

The features tensor dominates. Either drop `MONO_NUM_EXTRACT_TOKENS_TEST`
or `MONO_EXPANSION_FACTOR`. Both scale `features.npy` linearly:

| Setting change                                  | Features.npy size |
|-------------------------------------------------|-------------------|
| Defaults                                        | 245 GB            |
| `MONO_NUM_EXTRACT_TOKENS_TEST=500_000` (½)      | 122 GB            |
| `MONO_EXPANSION_FACTOR=32` (½)                  | 122 GB            |
| Both halved                                     |  61 GB            |

### "I want a faster smoke test"

```bash
MONO_NUM_GPUS=1 \
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
MONO_ANALYSIS_MIN_DISTINCT_SEQUENCES=3
```

End-to-end in well under 10 minutes including the model download (or 1 min
if the model is cached).

### "I want infer.py to fit on less GPU"

- Halve `MONO_INFER_FEATURE_BLOCK` — direct memory saving in the per-block
  intermediate.
- The activations buffer is the next-largest GPU allocation (~4 GB bf16);
  there's no knob to chunk it, but it's already conservative.

### "I want analyse.py to finish faster"

LLM calls dominate. Lower `MONO_ANALYSIS_NUM_FEATURES` or, if you don't
need every feature characterised, lower `MONO_LLM_NUM_EXAMPLES` so each
prompt is shorter (still ~3 req/s on the server but each request is
cheaper for the model).
