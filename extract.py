#!/usr/bin/env python3
"""
Extract residual-stream activations from the target layer of the model.

Two disjoint splits are produced in a single forward pass over the dataset:

    train:  NUM_EXTRACT_TOKENS_TRAIN tokens   ->  ACTIVATIONS_TRAIN_DIR
    test:   NUM_EXTRACT_TOKENS_TEST tokens    ->  ACTIVATIONS_TEST_DIR

Activations are saved *raw* (no normalisation applied).  We compute a single
"global" normalisation scalar from the train activations and store it inside
each split's meta.json.  Downstream consumers (train.py, infer.py) apply the
scalar at load time so that the same SAE can be evaluated on raw activations
later.

Per-split outputs:
    activations.npy   – float32, shape (num_sequences, SEQ_LEN, d_model)  (raw)
    token_ids.npy     – int32,   shape (num_sequences, SEQ_LEN)
    sequences.jsonl   – one JSON line per sequence:
                          {"text": "...", "tokens": ["t0", "t1", ...]}
    meta.json         – split metadata (incl. shared `scale`)
"""

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
    NUM_EXTRACT_TOKENS_TRAIN,
    NUM_EXTRACT_TOKENS_TEST,
    NUM_GPUS,
    GPU_IDS,
    ACTIVATIONS_TRAIN_DIR,
    ACTIVATIONS_TEST_DIR,
)

# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _build_max_memory() -> dict[int, str]:
    """Build a max_memory dict that restricts loading to the configured GPUs."""
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
    prefix_len = 10  # magic(6) + version(2) + HEADER_LEN(2)
    pad = 64 - ((prefix_len + len(header_dict) + 1) % 64)
    header_dict += " " * pad + "\n"
    hdr_len = len(header_dict)
    return (
        b"\x93NUMPY"
        + struct.pack("<BB", 1, 0)
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
            chunk = raw.read(4 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(npy_path)
    raw_path.unlink()


# ---------------------------------------------------------------------------
# Dataset iteration
# ---------------------------------------------------------------------------

def _iter_sequences(tokenizer):
    """Yield successive SEQ_LEN-token sequences as plain Python lists.

    Documents are tokenised on-the-fly and concatenated into a single stream
    that is then chopped into fixed-length windows.  No padding, no overlap.
    The iterator runs forever (or until the dataset is exhausted), so the
    caller is expected to take only as many sequences as needed.
    """
    data_path = DATASET_DIR / "data"
    shard_files = sorted(data_path.glob("shard_*.jsonl"))
    if not shard_files:
        raise FileNotFoundError(f"No shard files in {data_path}. Run download.py first.")

    token_buffer: list[int] = []
    for shard_file in shard_files:
        with open(shard_file) as f:
            for line in f:
                row = json.loads(line)
                text = row.get(DATASET_TEXT_FIELD, "")
                if not text:
                    continue
                token_buffer.extend(tokenizer.encode(text, add_special_tokens=False))
                while len(token_buffer) >= SEQ_LEN:
                    yield token_buffer[:SEQ_LEN]
                    token_buffer = token_buffer[SEQ_LEN:]


def _iter_token_batches(seq_iter, max_sequences: int, seqs_per_batch: int):
    """Yield fixed-size batches of sequences from *seq_iter*, up to *max_sequences*.

    Uses ``next()`` rather than ``for`` so that no extra sequence is drawn
    from the iterator past *max_sequences* (important when the same iterator
    is shared between the train and test splits).
    """
    batch: list[list[int]] = []
    yielded = 0
    while yielded < max_sequences:
        try:
            seq = next(seq_iter)
        except StopIteration:
            break
        batch.append(seq)
        yielded += 1
        if len(batch) >= seqs_per_batch:
            yield torch.tensor(batch, dtype=torch.long)
            batch = []
    if batch:
        yield torch.tensor(batch, dtype=torch.long)


# ---------------------------------------------------------------------------
# Per-split extraction
# ---------------------------------------------------------------------------

def _extract_split(
    split_name: str,
    out_dir: Path,
    target_tokens: int,
    seq_iter,
    model,
    tokenizer,
    target_layer,
    d_model: int,
    seqs_per_batch: int,
):
    """Extract activations for a single split.  Returns (sum_sq_norms, n_tokens, n_seqs)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    act_raw = out_dir / "activations.raw"
    tok_raw = out_dir / "token_ids.raw"
    txt_path = out_dir / "sequences.jsonl"
    act_path = out_dir / "activations.npy"
    tok_path = out_dir / "token_ids.npy"

    n_target_seqs = target_tokens // SEQ_LEN
    if n_target_seqs == 0:
        raise ValueError(
            f"[extract] {split_name}: target_tokens={target_tokens} < SEQ_LEN={SEQ_LEN}"
        )

    captured: list[torch.Tensor] = []

    def _hook_fn(_module, _input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden.detach().float().cpu())

    hook = target_layer.register_forward_hook(_hook_fn)

    sum_sq_norms = 0.0
    total_tokens = 0
    total_seqs = 0

    pbar = tqdm(total=n_target_seqs * SEQ_LEN, desc=f"extract[{split_name}]", unit="tok")
    act_file = open(act_raw, "wb")
    tok_file = open(tok_raw, "wb")
    txt_file = open(txt_path, "w")

    try:
        for batch_ids in _iter_token_batches(seq_iter, n_target_seqs,
                                              seqs_per_batch):
            captured.clear()
            with torch.no_grad():
                _ = model(input_ids=batch_ids.to(model.device))
            del _

            acts = captured[0]  # (B, SEQ_LEN, d_model)  cpu fp32
            B = acts.shape[0]

            # Write activations as raw bytes — same byte layout for both
            # (B, SEQ_LEN, d) and (B*SEQ_LEN, d), so the .npy header can pick.
            acts_np = acts.numpy()
            act_file.write(acts_np.tobytes())
            sum_sq_norms += float((acts_np ** 2).sum())

            tok_np = batch_ids.numpy().astype(np.int32)
            tok_file.write(tok_np.tobytes())

            for seq_idx in range(B):
                ids = batch_ids[seq_idx].tolist()
                text = tokenizer.decode(ids, skip_special_tokens=False)
                tokens = [tokenizer.decode([tid]) for tid in ids]
                txt_file.write(json.dumps({"text": text, "tokens": tokens}) + "\n")

            total_seqs += B
            total_tokens += B * SEQ_LEN
            pbar.update(B * SEQ_LEN)

            if total_seqs >= n_target_seqs:
                break
    finally:
        pbar.close()
        hook.remove()
        act_file.close()
        tok_file.close()
        txt_file.close()

    print(f"[extract] {split_name}: collected {total_seqs:,} sequences "
          f"({total_tokens:,} tokens)")

    # Convert the raw streams to .npy with their final 3D / 2D shapes.
    _write_npy_from_raw(act_raw, act_path,
                        np.dtype(np.float32), (total_seqs, SEQ_LEN, d_model))
    _write_npy_from_raw(tok_raw, tok_path,
                        np.dtype(np.int32), (total_seqs, SEQ_LEN))

    return sum_sq_norms, total_tokens, total_seqs


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def extract_activations():
    """Extract train + test splits in a single sweep over the dataset."""
    train_meta_path = ACTIVATIONS_TRAIN_DIR / "meta.json"
    test_meta_path = ACTIVATIONS_TEST_DIR / "meta.json"
    if train_meta_path.exists() and test_meta_path.exists():
        print(f"[extract] Activations already exist at "
              f"{ACTIVATIONS_TRAIN_DIR.parent}, skipping.")
        return

    ACTIVATIONS_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVATIONS_TEST_DIR.mkdir(parents=True, exist_ok=True)

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

    target_layer = model.model.layers[LAYER_INDEX]

    # 4 sequences per batch keeps batch_tokens ≈ 2k as in the original code.
    seqs_per_batch = max(1, 2048 // SEQ_LEN)

    # Single shared iterator → train and test draw disjoint sequences.
    seq_iter = _iter_sequences(tokenizer)

    # ---- Train split ------------------------------------------------------
    train_sum_sq, train_tok, train_seqs = _extract_split(
        "train", ACTIVATIONS_TRAIN_DIR, NUM_EXTRACT_TOKENS_TRAIN,
        seq_iter, model, tokenizer, target_layer, d_model, seqs_per_batch,
    )

    # ---- Test split (continues from where train left off) ----------------
    test_sum_sq, test_tok, test_seqs = _extract_split(
        "test", ACTIVATIONS_TEST_DIR, NUM_EXTRACT_TOKENS_TEST,
        seq_iter, model, tokenizer, target_layer, d_model, seqs_per_batch,
    )

    # ---- Free model -------------------------------------------------------
    del model
    torch.cuda.empty_cache()

    # ---- Compute global normalisation scalar (from TRAIN only) ------------
    mean_sq_norm = train_sum_sq / train_tok
    scale = math.sqrt(d_model / mean_sq_norm)
    print(f"[extract] Normalisation scale = {scale:.6f}  "
          f"(train mean ||x||^2 = {mean_sq_norm:.1f}, target = {d_model})")

    # ---- Write metadata ---------------------------------------------------
    common = {
        "model_name": MODEL_NAME,
        "layer_index": LAYER_INDEX,
        "d_model": d_model,
        "seq_len": SEQ_LEN,
        "scale": scale,
        "act_file": "activations.npy",
        "token_ids_file": "token_ids.npy",
        "sequences_file": "sequences.jsonl",
    }
    train_meta_path.write_text(json.dumps(
        {**common, "split": "train",
         "num_sequences": int(train_seqs),
         "num_tokens": int(train_tok)},
        indent=2,
    ))
    test_meta_path.write_text(json.dumps(
        {**common, "split": "test",
         "num_sequences": int(test_seqs),
         "num_tokens": int(test_tok)},
        indent=2,
    ))
    print(f"[extract] Done. train: {train_tok:,} tokens, "
          f"test: {test_tok:,} tokens.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    extract_activations()


if __name__ == "__main__":
    main()
