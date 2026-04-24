#!/usr/bin/env python3
"""
Extract residual-stream activations from the target layer of the model.

Workflow
--------
1. Load the model (fp16) across the configured GPUs.
2. Tokenize dataset examples into fixed-length sequences of SEQ_LEN tokens.
3. Run each batch through the model (no grad) and capture the residual-stream
   output at LAYER_INDEX via a forward hook.
4. Activations are streamed to raw binary files on disk so that RAM usage
   stays bounded regardless of NUM_EXTRACT_TOKENS.
5. After all activations are collected, a .npy header is prepended and the
   normalization scalar (E[||x||^2] = D_MODEL) is applied in-place via
   chunk-wise memmap access.
6. Token IDs and decoded text are saved alongside the activations to support
   downstream interpretability analysis.

Output files (in ACTIVATIONS_DIR):
    activations.npy      – float32, shape (N, d_model), normalised
    token_ids.npy        – int32,  shape (num_sequences, SEQ_LEN)
    texts.jsonl          – one JSON line per sequence: {"text": "..."}
    shuffle_indices.npy  – int64,  shape (N,), pre-shuffled for training
    meta.json            – extraction metadata
"""

import ctypes
import ctypes.util
import io
import json
import math
import struct
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from constants import (
    MODEL_NAME,
    MODEL_DIR,
    DATASET_DIR,
    DATASET_TEXT_FIELD,
    LAYER_INDEX,
    SEQ_LEN,
    NUM_EXTRACT_TOKENS,
    NUM_GPUS,
    GPU_IDS,
    ACTIVATIONS_DIR,
)

# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _build_max_memory() -> dict[int, str]:
    """Build a max_memory dict that restricts loading to the configured GPUs.

    This lets transformers/accelerate compute the device_map automatically
    (correctly handling tied weights like embed_tokens / lm_head) while
    keeping weights off any GPUs we don't want to use (e.g. the 3060).
    """
    mem: dict[int, str] = {}
    for gid in GPU_IDS:
        total = torch.cuda.get_device_properties(gid).total_memory
        usable = max(0, total - 1 * 1024**3)
        mem[gid] = f"{usable // 1024**2}MiB"
    return mem


# ---------------------------------------------------------------------------
# Low-level helpers for writing .npy without memmap
# ---------------------------------------------------------------------------

def _npy_header_bytes(dtype: np.dtype, shape: tuple[int, ...]) -> bytes:
    """Build the bytes for a NumPy .npy v1.0 header."""
    descr = np.lib.format.dtype_to_descr(dtype)
    header_dict = f"{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}"
    # Pad to 64-byte alignment (header = magic(6) + version(2) + HEADER_LEN(2) + payload).
    prefix_len = 10  # 6 + 2 + 2
    pad = 64 - ((prefix_len + len(header_dict) + 1) % 64)  # +1 for '\n'
    header_dict += " " * pad + "\n"
    hdr_len = len(header_dict)
    return (
        b"\x93NUMPY"          # magic
        + struct.pack("<BB", 1, 0)  # version 1.0
        + struct.pack("<H", hdr_len)
        + header_dict.encode("latin-1")
    )


def _write_npy_from_raw(raw_path: Path, npy_path: Path,
                         dtype: np.dtype, shape: tuple[int, ...]):
    """Prepend a .npy header to a raw binary file, producing a valid .npy."""
    header = _npy_header_bytes(dtype, shape)
    tmp = npy_path.with_suffix(".npy.tmp")
    with open(tmp, "wb") as out, open(raw_path, "rb") as raw:
        out.write(header)
        while True:
            chunk = raw.read(4 * 1024 * 1024)  # 4 MiB
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(npy_path)
    raw_path.unlink()


def _drop_page_cache(mm: np.memmap, start_row: int, end_row: int):
    """Advise the kernel to drop pages for rows [start_row, end_row).

    Uses posix_fadvise(POSIX_FADV_DONTNEED) on the byte range so that
    written/read pages are evicted from the page cache immediately.
    Falls back to a no-op on platforms where fadvise is unavailable.
    """
    try:
        libc_name = ctypes.util.find_library("c")
        if libc_name is None:
            return
        libc = ctypes.CDLL(libc_name, use_errno=True)
        POSIX_FADV_DONTNEED = 4
        row_bytes = mm.strides[0]
        offset = mm.offset + start_row * row_bytes
        length = (end_row - start_row) * row_bytes
        # mm._mmap is the underlying mmap object; fileno() gives the fd.
        fd = mm._mmap.fileno() if hasattr(mm, "_mmap") else -1
        if fd < 0:
            return
        libc.posix_fadvise(fd, ctypes.c_long(offset),
                           ctypes.c_long(length), POSIX_FADV_DONTNEED)
    except Exception:
        pass  # best-effort


# ---------------------------------------------------------------------------
# Dataset iteration
# ---------------------------------------------------------------------------

def _iter_token_batches(tokenizer, batch_tokens: int = 2048):
    """Yield batches of token IDs from the downloaded dataset shards.

    Each yielded tensor has shape (B, SEQ_LEN) with B up to
    ``batch_tokens // SEQ_LEN``.  We concatenate all document tokens into a
    single stream and chop into fixed-length windows (no padding waste).
    """
    data_path = DATASET_DIR / "data"
    shard_files = sorted(data_path.glob("shard_*.jsonl"))
    if not shard_files:
        raise FileNotFoundError(f"No shard files in {data_path}. Run download.py first.")

    seqs_per_batch = max(1, batch_tokens // SEQ_LEN)
    token_buffer: list[int] = []
    seq_buffer: list[list[int]] = []
    total_tokens = 0

    for shard_file in shard_files:
        with open(shard_file) as f:
            for line in f:
                row = json.loads(line)
                text = row.get(DATASET_TEXT_FIELD, "")
                if not text:
                    continue
                ids = tokenizer.encode(text, add_special_tokens=False)
                token_buffer.extend(ids)

                while len(token_buffer) >= SEQ_LEN:
                    seq_buffer.append(token_buffer[:SEQ_LEN])
                    token_buffer = token_buffer[SEQ_LEN:]
                    total_tokens += SEQ_LEN

                    if len(seq_buffer) >= seqs_per_batch:
                        yield torch.tensor(seq_buffer, dtype=torch.long)
                        seq_buffer = []

                    if total_tokens >= NUM_EXTRACT_TOKENS:
                        if seq_buffer:
                            yield torch.tensor(seq_buffer, dtype=torch.long)
                        return

    if seq_buffer:
        yield torch.tensor(seq_buffer, dtype=torch.long)


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def extract_activations():
    """Run the full extraction pipeline."""
    ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)

    meta_path = ACTIVATIONS_DIR / "meta.json"
    act_path = ACTIVATIONS_DIR / "activations.npy"
    tok_path = ACTIVATIONS_DIR / "token_ids.npy"
    txt_path = ACTIVATIONS_DIR / "texts.jsonl"
    idx_path = ACTIVATIONS_DIR / "shuffle_indices.npy"

    if meta_path.exists():
        print(f"[extract] Activations already exist at {ACTIVATIONS_DIR}, skipping.")
        return

    # Temporary raw binary files (no .npy header → no memmap during writes).
    act_raw = ACTIVATIONS_DIR / "activations.raw"
    tok_raw = ACTIVATIONS_DIR / "token_ids.raw"

    # ---- Load model -------------------------------------------------------
    print(f"[extract] Loading model {MODEL_NAME} ...")

    if NUM_GPUS == 1:
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR),
            dtype=torch.float16,
            device_map={"": GPU_IDS[0]},
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR),
            dtype=torch.float16,
            device_map="auto",
            max_memory=_build_max_memory(),
        )
    model.eval()

    d_model = model.config.hidden_size
    print(f"[extract] Using layer {LAYER_INDEX}, d_model={d_model}")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Register hook on the target layer --------------------------------
    captured: list[torch.Tensor] = []

    def _hook_fn(_module, _input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden.detach().float().cpu())

    target_layer = model.model.layers[LAYER_INDEX]
    hook = target_layer.register_forward_hook(_hook_fn)

    # ---- Forward pass — stream to raw binary files ------------------------
    max_tokens = NUM_EXTRACT_TOKENS
    total_tokens_written = 0
    total_seqs_written = 0
    sum_sq_norms = 0.0

    pbar = tqdm(total=max_tokens, desc="extracting", unit="tok")
    act_file = open(act_raw, "wb")
    tok_file = open(tok_raw, "wb")
    txt_file = open(txt_path, "w")

    for batch_ids in _iter_token_batches(tokenizer):
        captured.clear()

        with torch.no_grad():
            _ = model(input_ids=batch_ids.to(model.device))
        # Free the model output (logits, KV cache) immediately.
        del _

        acts = captured[0]              # (B, SEQ_LEN, d_model)
        B = acts.shape[0]
        acts_flat = acts.reshape(-1, d_model)  # (B*SEQ_LEN, d)
        n_tok = acts_flat.shape[0]

        # Trim if this batch would overshoot.
        remaining = max_tokens - total_tokens_written
        if n_tok > remaining:
            trim_seqs = remaining // SEQ_LEN
            if trim_seqs == 0:
                break
            B = trim_seqs
            n_tok = B * SEQ_LEN
            acts_flat = acts_flat[:n_tok]
            batch_ids = batch_ids[:B]

        # Convert to numpy and write raw bytes — no memmap, no page-cache
        # residency growth.
        acts_np = acts_flat.numpy()                        # float32 view
        act_file.write(acts_np.tobytes())

        sum_sq_norms += float((acts_flat ** 2).sum())

        tok_np = batch_ids.numpy().astype(np.int32)
        tok_file.write(tok_np.tobytes())

        for seq_idx in range(B):
            text = tokenizer.decode(batch_ids[seq_idx].tolist(),
                                    skip_special_tokens=False)
            txt_file.write(json.dumps({"text": text}) + "\n")

        total_tokens_written += n_tok
        total_seqs_written += B
        pbar.update(n_tok)

        # Explicitly free batch tensors.
        del acts, acts_flat, acts_np, tok_np

        if total_tokens_written >= max_tokens:
            break

    pbar.close()
    hook.remove()
    act_file.close()
    tok_file.close()
    txt_file.close()

    N = total_tokens_written
    N_seqs = total_seqs_written
    print(f"[extract] Collected {N:,} activation vectors "
          f"({N_seqs:,} sequences of {SEQ_LEN} tokens)")

    # ---- Convert raw binary → .npy ---------------------------------------
    print("[extract] Writing .npy files ...")
    _write_npy_from_raw(act_raw, act_path,
                        np.dtype(np.float32), (N, d_model))
    _write_npy_from_raw(tok_raw, tok_path,
                        np.dtype(np.int32), (N_seqs, SEQ_LEN))

    # ---- Free the model to reclaim GPU memory before normalisation --------
    del model
    torch.cuda.empty_cache()

    # ---- Normalize in-place via memmap (chunk-wise) -----------------------
    mean_sq_norm = sum_sq_norms / N
    scale = math.sqrt(d_model / mean_sq_norm)
    print(f"[extract] Normalization scale = {scale:.6f}  "
          f"(mean ||x||^2 before: {mean_sq_norm:.1f}, target: {d_model})")
    print("[extract] Applying normalisation ...")

    act_mm = np.load(str(act_path), mmap_mode="r+")
    chunk = 8192
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        act_mm[start:end] *= scale
        act_mm.flush()
        _drop_page_cache(act_mm, start, end)
    del act_mm

    # ---- Shuffled indices for training ------------------------------------
    print("[extract] Generating shuffled indices ...")
    shuffle_idx = np.random.permutation(N).astype(np.int64)
    np.save(str(idx_path), shuffle_idx)

    # ---- Metadata ---------------------------------------------------------
    meta = {
        "model_name": MODEL_NAME,
        "layer_index": LAYER_INDEX,
        "d_model": d_model,
        "seq_len": SEQ_LEN,
        "num_tokens": int(N),
        "num_sequences": int(N_seqs),
        "scale": scale,
        "act_file": act_path.name,
        "token_ids_file": tok_path.name,
        "texts_file": txt_path.name,
        "shuffle_file": idx_path.name,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[extract] Done. {N:,} vectors saved.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    extract_activations()


if __name__ == "__main__":
    main()
