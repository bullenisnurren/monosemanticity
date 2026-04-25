# Monosemanticity: Sparse Autoencoder for LLM Interpretability

This repository contains an implementation of the sparse autoencoder (SAE) approach from the paper [Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet](https://transformer-circuits.pub/2024/scaling-monosemanticity) by Anthropic. The pipeline extracts, trains, and analyses SAEs on transformer model activations to discover interpretable "features" - linear directions in activation space that correspond to human-understandable concepts.

## Overview

The approach is based on two key hypotheses:
- **Linear Representation Hypothesis**: Neural networks represent meaningful concepts as directions (linear subspaces) in their activation spaces.
- **Superposition Hypothesis**: Networks use the high dimensionality of their representations to encode more features than there are dimensions, through nearly-orthogonal "superposed" directions.

Sparse autoencoders decompose entangled model activations into cleaner, more interpretable components called **features**.

## Pipeline

```
download.py  ──►  extract.py  ──►  train.py  ──►  analyse.py
   │              │             │             │
   ▼              ▼             ▼             ▼
Model &      Activations     SAE          Metrics &
Dataset     (residuals)    Checkpoints  Interpretations
```

### 1. download.py
Downloads the target language model and dataset to local storage.

```bash
python download.py
```

- **Model**: Loaded from HuggingFace (default: `meta-llama/Llama-3.2-1B`)
- **Dataset**: Streaming download from HuggingFace Datasets (default: `monology/pile-uncopyrighted`)
- Saves to `./data/models/<model_slug>/` and `./data/datasets/<dataset_slug>/`

### 2. extract.py
Extracts residual-stream activations from a specific transformer layer.

```bash
python extract.py
```

Workflow:
1. Load the model (fp16) across configured GPUs
2. Tokenize dataset into fixed-length sequences (`SEQ_LEN=512` tokens)
3. Run inference and capture residual-stream output at `LAYER_INDEX` via forward hook
4. Stream activations to binary files (memory-mapped for bounded RAM)
5. Normalize so E[||x||²] = d_model (residual stream dimension)
6. Save pre-shuffled indices for training

Output files (in `./data/activations/<model_slug>/layer<N>/`):
- `activations.npy` — float32, shape (N, d_model), normalized
- `token_ids.npy` — int32, shape (num_sequences, SEQ_LEN)
- `texts.jsonl` — one JSON line per sequence: `{"text": "..."}`
- `shuffle_indices.npy` — int64, pre-shuffled indices
- `meta.json` — extraction metadata

### 3. train.py
Trains a sparse autoencoder on the extracted activations.

```bash
python train.py
```

**Architecture** (from Scaling Monosemanticity paper):
```
encode:  f = ReLU( W_enc @ (x - b_dec) + b_enc )
decode:  x_hat = W_dec @ f + b_dec
loss:   MSE(x, x_hat) + λ * Σᵢ fᵢ * ||w_decᵢ||₂
```

Where:
- `W_dec ∈ ℝ^(d_model × dict_size)` — decoder weights
- `W_enc ∈ ℝ^(dict_size × d_model)` — encoder weights  
- `f` — feature activations (sparse, due to ReLU)
- `λ` — L1 coefficient (promotes sparsity)

**Key training details**:
- Adam optimizer (no weight decay)
- L1 coefficient linearly warmed up over first 5% of steps
- Learning rate linearly decayed to 0 over last 20% of steps
- Gradient norm clipped to 1.0
- Decoder columns initialized as random directions with fixed norm
- Encoder initialized as transpose of decoder

**Hyperparameters** (defaults, configurable via env vars):
| Parameter | Default | Description |
|-----------|---------|-------------|
| EXPANSION_FACTOR | 64 | Dictionary size = d_model × expansion |
| NUM_TRAINING_STEPS | 100,000 | Training iterations |
| BATCH_SIZE | 4,096 | Batch size |
| L1_COEFF | 2.0 | Sparsity penalty |
| LR | 5e-5 | Learning rate |
| DECODER_INIT_NORM | 0.1 | Initial decoder column norm |

**Output**: Checkpoints saved to `./data/sae_checkpoints/sae_step_*.pt`

### 4. analyse.py
Analyses a trained SAE checkpoint.

```bash
python analyse.py
```

Computes:
- **L0**: Mean number of active features per token
- **Explained Variance**: How much activation variance the SAE captures
- **Dead Features**: Features that never activate on the test set
- **Feature Frequencies**: How often each feature fires
- **Max Activations**: Peak activation values per feature
- **Top-k Examples**: Highest-activating token indices for sampled features
- **Decoder Neighborhoods**: Cosine similarity between decoder directions

Output: `./data/analysis/report.json`

## Architecture Details

### Sparse Autoencoder

The SAE decomposes model activations into:
```
x = Σᵢ fᵢ * dᵢ + b_dec
```

where:
- `dᵢ = W_dec[:,i] / ||W_dec[:,i]||₂` — unit-norm decoder direction (feature)
- `fᵢ` — feature activation coefficient
- `b_dec` — decoder bias

The L1 penalty on `fᵢ * ||W_dec[:,i]||₂` ensures sparsity while allowing decoder vectors to have arbitrary norms during training.

### Post-training Normalization

After training, decoder columns are normalized to unit norm:
```
W_enc'[i] = W_enc[i] * ||W_dec[:,i]||
W_dec'[:,i] = W_dec[:,i] / ||W_dec[:,i]||
```

This is mathematically equivalent but gives activations in "true" units.

## Configuration

All parameters can be configured via environment variables with the `MONO_` prefix:

```bash
# Model & Dataset
export MONO_MODEL_NAME="meta-llama/Llama-3.2-1B"
export MONO_DATASET_NAME="monology/pile-uncopyrighted"

# Extraction
export MONO_LAYER_INDEX=8          # Which layer to extract from
export MONO_SEQ_LEN=512           # Sequence length
export MONO_NUM_EXTRACT_TOKENS=10000000  # Tokens to extract

# SAE Architecture  
export MONO_EXPANSION_FACTOR=64    # Dictionary size multiplier

# Training
export MONO_NUM_TRAINING_STEPS=100000
export MONO_BATCH_SIZE=4096
export MONO_L1_COEFF=2.0
export MONO_LR=5e-5

# Hardware
export MONO_NUM_GPUS=2
export MONO_GPU_IDS="0,1"

# Analysis
export MONO_ANALYSIS_NUM_TOKENS=1000000
export MONO_ANALYSIS_SAMPLE_FEATURES=256
```

## Requirements

```
torch
transformers
huggingface_hub
datasets
numpy
tqdm
```

Install via:
```bash
pip install -r requirements.txt
```

## File Structure

```
.
├── download.py           # Download model & dataset
├── extract.py           # Extract activations
├── train.py            # Train SAE
├── analyse.py          # Analyse checkpoint
├── constants.py        # Configuration
├── requirements.txt    # Dependencies
└── data/
    ├── activations/   # Extracted activations
    ├── analysis/      # Analysis reports
    ├── datasets/      # Downloaded datasets
    ├── models/        # Downloaded models
    ├── sae_checkpoints/ # Trained SAE checkpoints
    └── paper.html     # Original Anthropic paper
```

## References

- [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity) (Anthropic, 2024)
- [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features) (Anthropic, 2023)
- [Superposition](https://transformer-circuits.pub/2022/toy_model) (Elhage et al., 2022)

