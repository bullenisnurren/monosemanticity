#!/usr/bin/env python3
"""
Extract residual-stream activations from the target layer of the model.

Workflow
--------
1. Load the model (fp16) across the configured GPUs with an explicit device map.
2. Tokenize dataset examples into fixed-length sequences of SEQ_LEN tokens.
3. Run each batch through the model (no grad) and capture the residual-stream
   output at LAYER_INDEX via a forward hook.
4. After all activations are collected, compute a single normalization scalar
   so that  E[||x||^2] = D_MODEL  (paper prescription).
5. Save the (normalised) activations as a memory-mapped float32 .npy file,
   a shuffled-index array, and a metadata JSON file.
"""

import json
import math
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
    D_MODEL,
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
        # Leave ~1 GB headroom for CUDA context / fragmentation.
        usable = max(0, total - 1 * 1024**3)
        mem[gid] = f"{usable // 1024**2}MiB"
    return mem


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

                # Chop complete sequences out of the buffer.
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

    # Flush any remaining complete batch.
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
    idx_path = ACTIVATIONS_DIR / "shuffle_indices.npy"

    if meta_path.exists():
        print(f"[extract] Activations already exist at {ACTIVATIONS_DIR}, skipping.")
        return

    # ---- Load model -------------------------------------------------------
    print(f"[extract] Loading model {MODEL_NAME} ...")

    if NUM_GPUS == 1:
        # Single GPU — place everything on cuda:0.
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR),
            dtype=torch.float16,
            device_map={"": GPU_IDS[0]},
        )
    else:
        # Multi-GPU — let accelerate compute the split.  We provide
        # max_memory to restrict it to our configured GPUs only.
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR),
            dtype=torch.float16,
            device_map="auto",
            max_memory=_build_max_memory(),
        )
    model.eval()

    # Verify d_model matches.
    actual_d = model.config.hidden_size
    if actual_d != D_MODEL:
        print(f"[extract] WARNING: D_MODEL={D_MODEL} but model has hidden_size={actual_d}. "
              f"Using {actual_d}.")

    d_model = actual_d

    print(f"[extract] Using layer {LAYER_INDEX}, d_model={d_model}")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Register hook on the target layer --------------------------------
    captured: list[torch.Tensor] = []

    def _hook_fn(_module, _input, output):
        # For Llama, each decoder layer returns (hidden_states, ...).
        hidden = output[0] if isinstance(output, tuple) else output
        # hidden: (batch, seq_len, d_model)  -- detach & move to CPU.
        captured.append(hidden.detach().float().cpu())

    target_layer = model.model.layers[LAYER_INDEX]
    hook = target_layer.register_forward_hook(_hook_fn)

    # ---- Forward pass over the dataset ------------------------------------
    total_tokens_extracted = 0
    pbar = tqdm(total=NUM_EXTRACT_TOKENS, desc="extracting", unit="tok")

    for batch_ids in _iter_token_batches(tokenizer):
        with torch.no_grad():
            # With device_map, accelerate hooks move tensors between devices
            # automatically.  We send input_ids to the device where the
            # outermost module's hook expects them.
            _ = model(input_ids=batch_ids.to(model.device))

        n_tok = batch_ids.shape[0] * batch_ids.shape[1]
        total_tokens_extracted += n_tok
        pbar.update(n_tok)

        if total_tokens_extracted >= NUM_EXTRACT_TOKENS:
            break

    pbar.close()
    hook.remove()

    # ---- Stack & reshape to (N, d_model) ----------------------------------
    print("[extract] Concatenating activations ...")
    all_acts = torch.cat(captured, dim=0)  # (total_seqs, SEQ_LEN, d_model)
    all_acts = all_acts.reshape(-1, d_model)  # (N, d_model)

    # Trim to exactly NUM_EXTRACT_TOKENS if we overshot.
    if all_acts.shape[0] > NUM_EXTRACT_TOKENS:
        all_acts = all_acts[:NUM_EXTRACT_TOKENS]

    N = all_acts.shape[0]
    print(f"[extract] Collected {N:,} activation vectors of dim {d_model}")

    # ---- Normalize: scale so E[||x||^2] = d_model ------------------------
    mean_sq_norm = (all_acts.norm(dim=1) ** 2).mean().item()
    scale = math.sqrt(d_model / mean_sq_norm)
    all_acts *= scale
    print(f"[extract] Normalization scale = {scale:.6f}  "
          f"(mean ||x||^2 before: {mean_sq_norm:.1f}, target: {d_model})")

    # ---- Save to disk as memory-mapped npy --------------------------------
    print(f"[extract] Saving activations to {act_path} ...")
    np_acts = all_acts.numpy()
    np.save(str(act_path), np_acts)

    # Pre-compute a shuffled index for training.
    print("[extract] Generating shuffled indices ...")
    shuffle_idx = np.random.permutation(N).astype(np.int64)
    np.save(str(idx_path), shuffle_idx)

    # Save metadata.
    meta = {
        "model_name": MODEL_NAME,
        "layer_index": LAYER_INDEX,
        "d_model": d_model,
        "seq_len": SEQ_LEN,
        "num_tokens": int(N),
        "scale": scale,
        "act_file": act_path.name,
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
