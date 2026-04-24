#!/usr/bin/env python3
"""
Download the model and dataset to local directories.

Model  -> ./data/models/<model_slug>/
Dataset -> ./data/datasets/<dataset_slug>/

The dataset download is limited to roughly the number of examples needed to
yield NUM_EXTRACT_TOKENS tokens (with a 1.5x safety margin to account for
short documents and tokenizer overhead).
"""

import math
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from constants import (
    MODEL_NAME,
    MODEL_DIR,
    DATASET_NAME,
    DATASET_DIR,
    DATASET_SPLIT,
    DATASET_TEXT_FIELD,
    NUM_EXTRACT_TOKENS,
    SEQ_LEN,
)


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

def download_model() -> Path:
    """Download (or verify) the HuggingFace model locally.

    We consider the download complete only when at least one ``.safetensors``
    or ``.bin`` weight file is present — config/tokenizer files alone are not
    enough.
    """
    weight_globs = list(MODEL_DIR.glob("*.safetensors")) + list(MODEL_DIR.glob("*.bin"))
    if weight_globs:
        print(f"[download] Model weights already present at {MODEL_DIR}, skipping.")
        return MODEL_DIR

    print(f"[download] Downloading model {MODEL_NAME} -> {MODEL_DIR} ...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_NAME,
        local_dir=str(MODEL_DIR),
    )
    print(f"[download] Model saved to {MODEL_DIR}")
    return MODEL_DIR


# ---------------------------------------------------------------------------
# Dataset download
# ---------------------------------------------------------------------------

def _estimate_examples_needed(tokenizer, target_tokens: int, seq_len: int) -> int:
    """Rough estimate of how many dataset examples we need.

    We assume each example yields on average ``seq_len`` usable tokens after
    tokenisation (conservative), then add a 1.5x safety margin.
    """
    return int(math.ceil(target_tokens / seq_len * 1.5))


def download_dataset() -> Path:
    """Stream-download enough of the dataset for extraction."""
    meta_path = DATASET_DIR / "meta.json"
    data_path = DATASET_DIR / "data"

    if meta_path.exists():
        print(f"[download] Dataset already present at {DATASET_DIR}, skipping.")
        return DATASET_DIR

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    data_path.mkdir(parents=True, exist_ok=True)

    # Estimate how many examples we need.
    print(f"[download] Loading tokenizer for {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    n_examples = _estimate_examples_needed(tokenizer, NUM_EXTRACT_TOKENS, SEQ_LEN)
    print(f"[download] Target: {NUM_EXTRACT_TOKENS:,} tokens  ->  "
          f"fetching ~{n_examples:,} examples (safety margin 1.5x)")

    # Stream the dataset so we only download what we need.
    print(f"[download] Streaming dataset {DATASET_NAME} (split={DATASET_SPLIT}) ...")
    ds_stream = load_dataset(
        DATASET_NAME,
        split=DATASET_SPLIT,
        streaming=True,
    )

    # Collect examples into shards of 50k rows each (Arrow files).
    shard_size = 50_000
    shard_idx = 0
    buffer = []
    total_saved = 0

    for example in tqdm(ds_stream, total=n_examples, desc="downloading"):
        buffer.append(example)
        if len(buffer) >= shard_size:
            _flush_shard(buffer, data_path, shard_idx)
            total_saved += len(buffer)
            buffer = []
            shard_idx += 1
        if total_saved + len(buffer) >= n_examples:
            break

    # Flush remaining.
    if buffer:
        _flush_shard(buffer, data_path, shard_idx)
        total_saved += len(buffer)

    meta = {
        "dataset_name": DATASET_NAME,
        "split": DATASET_SPLIT,
        "text_field": DATASET_TEXT_FIELD,
        "num_examples": total_saved,
        "num_shards": shard_idx + 1,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[download] Saved {total_saved:,} examples in {shard_idx + 1} shard(s) "
          f"to {DATASET_DIR}")
    return DATASET_DIR


def _flush_shard(rows: list[dict], data_path: Path, shard_idx: int):
    """Write a list of dicts to a JSON-Lines shard."""
    path = data_path / f"shard_{shard_idx:05d}.jsonl"
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    download_model()
    download_dataset()
    print("[download] Done.")


if __name__ == "__main__":
    main()
