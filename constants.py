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
D_MODEL: int = _env("MONO_D_MODEL", 2048, int)
SEQ_LEN: int = _env("MONO_SEQ_LEN", 512, int)
NUM_EXTRACT_TOKENS: int = _env("MONO_NUM_EXTRACT_TOKENS", 20_000_000, int)

ACTIVATIONS_DIR: Path = DATA_DIR / "activations" / _model_slug / f"layer{LAYER_INDEX}"

# ---------------------------------------------------------------------------
# SAE architecture
# ---------------------------------------------------------------------------

EXPANSION_FACTOR: int = _env("MONO_EXPANSION_FACTOR", 64, int)
DICT_SIZE: int = D_MODEL * EXPANSION_FACTOR  # number of SAE features

# ---------------------------------------------------------------------------
# Training hyper-parameters  (paper defaults where applicable)
# ---------------------------------------------------------------------------

NUM_TRAINING_STEPS: int = _env("MONO_NUM_TRAINING_STEPS", 100_000, int)
BATCH_SIZE: int = _env("MONO_BATCH_SIZE", 4096, int)

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
CHECKPOINT_EVERY: int = _env("MONO_CHECKPOINT_EVERY", 100, int)
LOG_EVERY: int = _env("MONO_LOG_EVERY", 10, int)

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

ANALYSIS_DIR: Path = DATA_DIR / "analysis"
ANALYSIS_NUM_TOKENS: int = _env("MONO_ANALYSIS_NUM_TOKENS", 1_000_000, int)
# Number of features to compute detailed stats for (top-k examples, neighborhoods).
ANALYSIS_SAMPLE_FEATURES: int = _env("MONO_ANALYSIS_SAMPLE_FEATURES", 256, int)
# Number of top-activating examples to store per sampled feature.
ANALYSIS_TOP_K: int = _env("MONO_ANALYSIS_TOP_K", 20, int)
