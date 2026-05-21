#!/usr/bin/env python3
"""
Compute SAE features on the held-out test activations and persist them to
disk in a layout that analyse.py can read efficiently per-feature.

Pipeline position:    train.py -> infer.py -> analyse.py

On-disk layout
--------------
``features.npy`` has shape ``(dict_size, num_test_sequences, seq_len)`` and
dtype ``float16``.  This is the same information as a logical
``(N, S, F)`` tensor — just stored feature-major, so that
``features[fid]`` is a single 2-MB contiguous read.  Storing it this way
lets analyse.py inspect a handful of features without scanning 200+ GB.

Method
------
Test activations easily fit in GPU memory (a few GB), so we load them once
and iterate over **feature blocks** instead of sequence chunks:

    for f_start, f_end in feature_blocks:
        block_feats = ReLU(x @ W_enc[f_start:f_end].T + b_enc[f_start:f_end])
        block_feats *= ||W_dec[:, f_start:f_end]||
        features[f_start:f_end] = block_feats.permute(2,0,1).contiguous()

Each block write is fully sequential on disk, so total time is essentially
``feature_tensor_size / disk_bandwidth``.

Side artefacts (small) written next to ``features.npy``:

    fire_count.npy      (F,)         int64
    max_per_seq.npy     (N, F)       float16
    argmax_per_seq.npy  (N, F)       int16
    token_ids.npy + sequences.jsonl  (copied from the test split)

analyse.py uses the side arrays for dead-feature detection and top-k
sequence discovery, then reads only the relevant rows from features.npy.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from constants import (
    DECODER_INIT_NORM,
    GPU_IDS,
    ACTIVATIONS_TEST_DIR,
    CHECKPOINT_DIR,
    FEATURES_DIR,
    FEATURE_DTYPE,
    INFER_CHUNK_SEQUENCES,
    INFER_FEATURE_BLOCK,
)
from train import SparseAutoencoder


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _find_latest_checkpoint() -> Path:
    ckpts = sorted(CHECKPOINT_DIR.glob("sae_step_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {CHECKPOINT_DIR}")
    return ckpts[-1]


def _load_sae(ckpt_path: Path, device: torch.device):
    """Returns (sae, ckpt_dict)."""
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    sae = SparseAutoencoder(ckpt["d_model"], ckpt["dict_size"], DECODER_INIT_NORM)
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.to(device).eval()
    return sae, ckpt


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def infer():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out_meta_path = FEATURES_DIR / "meta.json"
    out_feat_path = FEATURES_DIR / "features.npy"

    if out_meta_path.exists() and out_feat_path.exists():
        print(f"[infer] Features already exist at {FEATURES_DIR}, skipping.")
        return

    device = torch.device(f"cuda:{GPU_IDS[0]}" if torch.cuda.is_available() else "cpu")

    # ---- Load SAE ---------------------------------------------------------
    ckpt_path = _find_latest_checkpoint()
    print(f"[infer] Loading checkpoint: {ckpt_path.name}")
    sae, ckpt = _load_sae(ckpt_path, device)

    activation_scale = float(ckpt["activation_scale"])
    print(f"[infer] Activation scale = {activation_scale:.6f}")

    # Fold b_dec into b_enc so we don't have to allocate a centred copy of x:
    #   ReLU((x - b_dec) @ W_enc.T + b_enc)
    # = ReLU(x @ W_enc.T + (b_enc - b_dec @ W_enc.T)).
    with torch.no_grad():
        dec_norms = sae.W_dec.norm(dim=0).contiguous()              # (F,)
        b_enc_eff = (sae.b_enc - sae.b_dec @ sae.W_enc.T).contiguous()  # (F,)
        # Unit-norm decoder directions, laid out (F, d) so per-feature
        # reads in analyse.py are contiguous.
        W_dec_unit_fd = (sae.W_dec / dec_norms.clamp(min=1e-8)).T.contiguous()  # (F, d) fp32

    # Cast SAE to bf16 for inference.  bf16 matmul on Ampere is fast and the
    # precision is more than enough for interpretability work.
    sae = sae.to(torch.bfloat16)

    # ---- Locate test activations -----------------------------------------
    test_meta_path = ACTIVATIONS_TEST_DIR / "meta.json"
    if not test_meta_path.exists():
        raise FileNotFoundError(
            f"No test meta.json at {test_meta_path}. Run extract.py first."
        )
    with open(test_meta_path) as f:
        test_meta = json.load(f)

    test_acts_path = ACTIVATIONS_TEST_DIR / test_meta["act_file"]
    test_acts_mm = np.load(str(test_acts_path), mmap_mode="r")  # (N, S, d) fp32
    n_seqs, seq_len, d_model = test_acts_mm.shape
    dict_size = int(sae.dict_size)

    if d_model != sae.d_model:
        raise RuntimeError(
            f"d_model mismatch: activations={d_model} vs SAE={sae.d_model}"
        )

    print(f"[infer] Test activations: {n_seqs:,} sequences × {seq_len} tokens "
          f"× {d_model} dims ({n_seqs * seq_len * d_model * 4 / 1e9:.2f} GB "
          f"on disk, streaming).")

    # ---- Allocate output memmap (feature-major) --------------------------
    feat_dtype = np.dtype(FEATURE_DTYPE)
    feat_bytes = dict_size * n_seqs * seq_len * feat_dtype.itemsize
    print(f"[infer] Allocating features memmap: "
          f"shape=({dict_size}, {n_seqs}, {seq_len}) [F, N, S], "
          f"dtype={feat_dtype}, size={feat_bytes / 1e9:.2f} GB")
    features = np.lib.format.open_memmap(
        str(out_feat_path),
        mode="w+",
        dtype=feat_dtype,
        shape=(dict_size, n_seqs, seq_len),
    )

    # ---- Side arrays (kept on CPU, written once at the end) --------------
    fire_count = np.zeros(dict_size, dtype=np.int64)
    max_per_seq = np.empty((n_seqs, dict_size), dtype=np.float16)
    argmax_per_seq = np.empty((n_seqs, dict_size), dtype=np.int16)

    if seq_len > 32767:
        raise ValueError("argmax_per_seq uses int16; seq_len must be <= 32767.")

    # ---- Feature-block × sequence-chunk sweep ----------------------------
    #
    # For each feature block we walk the test set in chunks of
    # INFER_CHUNK_SEQUENCES sequences, building the block's features in a
    # CPU buffer one chunk at a time, then writing the whole block to the
    # memmap in one contiguous slab.  This keeps GPU memory O(chunk · S · d)
    # — independent of the test-set size — while preserving sequential disk
    # writes (one F-block per iteration, exactly as before).
    #
    # The test activations are read from a memmap, so on the first F-block
    # iteration the OS fills its page cache from disk and every subsequent
    # F-block reuses those cached pages for free.  Default settings give
    # ~17 GB of test activations, which sits comfortably in cache.
    f_block = max(1, INFER_FEATURE_BLOCK)
    s_chunk = max(1, INFER_CHUNK_SEQUENCES)
    n_blocks = (dict_size + f_block - 1) // f_block
    n_chunks = (n_seqs + s_chunk - 1) // s_chunk
    print(f"[infer] {n_blocks} feature blocks × {n_chunks} sequence chunks "
          f"(F_BLOCK={f_block}, CHUNK_SEQS={s_chunk}).")

    for bi in tqdm(range(n_blocks), desc="infer", unit="block"):
        f_start = bi * f_block
        f_end = min(f_start + f_block, dict_size)
        size = f_end - f_start

        with torch.no_grad():
            W_block = sae.W_enc[f_start:f_end]                       # (B, d) bf16
            b_block = b_enc_eff[f_start:f_end].to(torch.bfloat16)     # (B,)
            d_block = dec_norms[f_start:f_end].to(torch.bfloat16)     # (B,)

        # CPU staging buffer for this F-block's output.  Written once at
        # the end as a single contiguous slab into the memmap.
        f_buf = np.empty((size, n_seqs, seq_len), dtype=feat_dtype)

        for c_start in range(0, n_seqs, s_chunk):
            c_end = min(c_start + s_chunk, n_seqs)

            # CPU-side: read chunk from memmap (fp32), apply activation
            # scale, cast to bf16, then push to GPU.
            x_cpu = np.array(test_acts_mm[c_start:c_end], dtype=np.float32)
            x_cpu *= activation_scale
            x = torch.from_numpy(x_cpu).to(device, dtype=torch.bfloat16,
                                            non_blocking=True)

            with torch.no_grad():
                f_3d = torch.relu(x @ W_block.T + b_block)            # (chunk, S, B)
                f_3d.mul_(d_block)

                # Per-chunk × per-block contributions to the side arrays.
                fc_chunk = (f_3d > 0).sum(dim=(0, 1)).cpu().numpy()
                fire_count[f_start:f_end] += fc_chunk
                max_vals, max_idxs = f_3d.max(dim=1)                  # (chunk, B)
                max_per_seq[c_start:c_end, f_start:f_end] = (
                    max_vals.to(torch.float16).cpu().numpy())
                argmax_per_seq[c_start:c_end, f_start:f_end] = (
                    max_idxs.to(torch.int16).cpu().numpy())

                # (chunk, S, B) -> (B, chunk, S), fp16, contiguous for the
                # CPU buffer slice we're about to fill.
                f_out = (f_3d.permute(2, 0, 1).to(torch.float16)
                              .contiguous().cpu().numpy())
                del f_3d

            f_buf[:, c_start:c_end, :] = f_out
            del f_out, x, x_cpu

        # One contiguous write of the entire F-block.
        features[f_start:f_end] = f_buf
        del f_buf

    features.flush()

    # ---- Persist side arrays ---------------------------------------------
    np.save(str(FEATURES_DIR / "fire_count.npy"), fire_count)
    np.save(str(FEATURES_DIR / "max_per_seq.npy"), max_per_seq)
    np.save(str(FEATURES_DIR / "argmax_per_seq.npy"), argmax_per_seq)
    np.save(str(FEATURES_DIR / "decoder_directions.npy"),
            W_dec_unit_fd.to(torch.float16).cpu().numpy())

    for fname in ("token_ids.npy", "sequences.jsonl"):
        src = ACTIVATIONS_TEST_DIR / fname
        dst = FEATURES_DIR / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    # ---- Metadata ---------------------------------------------------------
    n_dead = int((fire_count == 0).sum())
    meta = {
        "checkpoint": ckpt_path.name,
        "step": int(ckpt["step"]),
        "activation_scale": activation_scale,
        "d_model": int(d_model),
        "dict_size": dict_size,
        "expansion_factor": int(ckpt["expansion_factor"]),
        "num_sequences": int(n_seqs),
        "seq_len": int(seq_len),
        "feature_dtype": str(feat_dtype),
        "feature_decoder_scaled": True,
        "features_layout": "feature_major",  # (F, N, S)  C-order
        "features_shape": [dict_size, int(n_seqs), int(seq_len)],
        "features_file": "features.npy",
        "fire_count_file": "fire_count.npy",
        "max_per_seq_file": "max_per_seq.npy",
        "argmax_per_seq_file": "argmax_per_seq.npy",
        "decoder_directions_file": "decoder_directions.npy",
        "token_ids_file": "token_ids.npy",
        "sequences_file": "sequences.jsonl",
        "test_split_meta": test_meta,
        "training_config": ckpt.get("config", {}),
    }
    out_meta_path.write_text(json.dumps(meta, indent=2))

    print(f"[infer] Wrote {out_feat_path}  "
          f"(features (F,N,S)={dict_size}×{n_seqs}×{seq_len})")
    print(f"[infer] Dead features (test set): {n_dead:,} / {dict_size:,} "
          f"({100 * n_dead / dict_size:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    infer()


if __name__ == "__main__":
    main()
