"""
Project constants for SAE training pipeline.

All values are read from environment variables with the MONO_ prefix.
If an env var is not set, a reasonable default is used (typically from the
"Scaling Monosemanticity" paper or adapted for Llama-3.2-1B on 2x RTX 3090).
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env(name: str, default, dtype=str):
    """Read an env var with a type cast, falling back to *default*."""
    val = os.environ.get(name)
    if val is None:
        return default
    if dtype is bool:
        return val.lower() in ("1", "true", "yes")
    return dtype(val)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# Model & dataset
# ---------------------------------------------------------------------------

MODEL_NAME: str = _env("MONO_MODEL_NAME", "meta-llama/Llama-3.2-1B")
DATASET_NAME: str = _env("MONO_DATASET_NAME", "monology/pile-uncopyrighted")
DATASET_SPLIT: str = _env("MONO_DATASET_SPLIT", "train")
DATASET_TEXT_FIELD: str = _env("MONO_DATASET_TEXT_FIELD", "text")

# Derived paths --  slashes in HF IDs are replaced with double underscores.
_model_slug = MODEL_NAME.replace("/", "__")
_dataset_slug = DATASET_NAME.replace("/", "__")

MODEL_DIR: Path = DATA_DIR / "models" / _model_slug
DATASET_DIR: Path = DATA_DIR / "datasets" / _dataset_slug

# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

LAYER_INDEX: int = _env("MONO_LAYER_INDEX", 8, int)
SEQ_LEN: int = _env("MONO_SEQ_LEN", 512, int)

# We extract two disjoint sets of activations: a (large) train set used to
# fit the SAE and a smaller held-out test set used for inference & analysis.
NUM_EXTRACT_TOKENS_TRAIN: int = _env("MONO_NUM_EXTRACT_TOKENS_TRAIN", 20_000_000, int)
NUM_EXTRACT_TOKENS_TEST: int = _env("MONO_NUM_EXTRACT_TOKENS_TEST", 1_000_000, int)

ACTIVATIONS_DIR: Path = DATA_DIR / "activations" / _model_slug / f"layer{LAYER_INDEX}"
ACTIVATIONS_TRAIN_DIR: Path = ACTIVATIONS_DIR / "train"
ACTIVATIONS_TEST_DIR: Path = ACTIVATIONS_DIR / "test"

# ---------------------------------------------------------------------------
# SAE architecture
# ---------------------------------------------------------------------------

EXPANSION_FACTOR: int = _env("MONO_EXPANSION_FACTOR", 64, int)

# ---------------------------------------------------------------------------
# Training hyper-parameters  (paper defaults where applicable)
# ---------------------------------------------------------------------------

NUM_TRAINING_STEPS: int = _env("MONO_NUM_TRAINING_STEPS", 200_000, int)
BATCH_SIZE: int = _env("MONO_BATCH_SIZE", 4096, int)

# Activation loader (train.py): we read whole sequences in random order from
# the on-disk memmap into a RAM buffer, then shuffle tokens inside the buffer
# before draining it batch-by-batch.  This avoids per-batch random seeks on
# disk storage while preserving near-token-level batch randomisation.
# Default buffer size: 2048 sequences ≈ 8 GB (for SEQ_LEN=512, d_model=2048).
BUFFER_SEQUENCES: int = _env("MONO_BUFFER_SEQUENCES", 2048, int)

L1_COEFF: float = _env("MONO_L1_COEFF", 5.0, float)
L1_WARMUP_FRAC: float = _env("MONO_L1_WARMUP_FRAC", 0.05, float)

LR: float = _env("MONO_LR", 5e-5, float)
LR_DECAY_FRAC: float = _env("MONO_LR_DECAY_FRAC", 0.20, float)

ADAM_BETA1: float = _env("MONO_ADAM_BETA1", 0.9, float)
ADAM_BETA2: float = _env("MONO_ADAM_BETA2", 0.999, float)

GRAD_CLIP_NORM: float = _env("MONO_GRAD_CLIP_NORM", 1.0, float)

DECODER_INIT_NORM: float = _env("MONO_DECODER_INIT_NORM", 0.1, float)

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------

NUM_GPUS: int = _env("MONO_NUM_GPUS", 2, int)
# When NUM_GPUS=1 we use cuda:0 only; when >1 we use cuda:0 .. cuda:{NUM_GPUS-1}.
GPU_IDS: list[int] = list(range(NUM_GPUS))

# ---------------------------------------------------------------------------
# Checkpointing & logging
# ---------------------------------------------------------------------------

CHECKPOINT_DIR: Path = DATA_DIR / "sae_checkpoints"
CHECKPOINT_EVERY: int = _env("MONO_CHECKPOINT_EVERY", 10_000, int)
LOG_EVERY: int = _env("MONO_LOG_EVERY", 10, int)

# ---------------------------------------------------------------------------
# Inference (infer.py)
# ---------------------------------------------------------------------------

# Where SAE features computed on the test activations are stored as a memmap.
FEATURES_DIR: Path = DATA_DIR / "features" / _model_slug / f"layer{LAYER_INDEX}"

# Stored feature dtype.  fp16 cuts disk usage in half with negligible loss
# for non-negative ReLU outputs.
FEATURE_DTYPE: str = _env("MONO_FEATURE_DTYPE", "float16")

# infer.py iterates over feature blocks of this size so it can write the
# (F, N, S) features tensor in fully sequential chunks (one contiguous block
# per feature group).  Larger = fewer disk seeks but more GPU memory.
INFER_FEATURE_BLOCK: int = _env("MONO_INFER_FEATURE_BLOCK", 512, int)

# ---------------------------------------------------------------------------
# Analysis (analyse.py)
# ---------------------------------------------------------------------------

ANALYSIS_DIR: Path = DATA_DIR / "analysis"

# How many non-dead features to randomly sample for the report.
ANALYSIS_NUM_FEATURES: int = _env("MONO_ANALYSIS_NUM_FEATURES", 100, int)
# Number of top-activating sequences to show per feature.
ANALYSIS_TOP_K: int = _env("MONO_ANALYSIS_TOP_K", 20, int)
# RNG seed for reproducible feature sampling.
ANALYSIS_SEED: int = _env("MONO_ANALYSIS_SEED", 0, int)

# ---------------------------------------------------------------------------
# LLM-based feature description (analyse.py)
# ---------------------------------------------------------------------------

# OpenAI-compatible endpoint base URL.
LLM_API_BASE_URL: str = _env("MONO_LLM_API_BASE_URL", "http://127.0.0.1:8000")
LLM_API_KEY: str = _env("MONO_LLM_API_KEY", "EMPTY")
LLM_API_MODEL: str = _env("MONO_LLM_API_MODEL", "empyrean")
LLM_API_TIMEOUT: float = _env("MONO_LLM_API_TIMEOUT", 60.0, float)
# How many top-k examples to include in the LLM prompt (≤ ANALYSIS_TOP_K).
LLM_NUM_EXAMPLES: int = _env("MONO_LLM_NUM_EXAMPLES", 10, int)
# Max characters per example sent to the LLM (truncated around the peak token).
LLM_EXAMPLE_CHARS: int = _env("MONO_LLM_EXAMPLE_CHARS", 240, int)
