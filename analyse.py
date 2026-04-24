#!/usr/bin/env python3
"""
Analyse a trained SAE checkpoint.

Computes:
  - L0  (mean active features per token)
  - Explained variance
  - Dead / alive feature counts
  - Per-feature activation frequency & max activation
  - Top-k activating token indices for a sample of features
  - Decoder cosine-similarity neighborhoods for a sample of features

Results are saved to ./data/analysis/ as a JSON report and printed to stdout.
"""

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from constants import (
    D_MODEL,
    EXPANSION_FACTOR,
    DECODER_INIT_NORM,
    NUM_GPUS,
    GPU_IDS,
    ACTIVATIONS_DIR,
    CHECKPOINT_DIR,
    ANALYSIS_DIR,
    ANALYSIS_NUM_TOKENS,
    ANALYSIS_SAMPLE_FEATURES,
    ANALYSIS_TOP_K,
)
from train import SparseAutoencoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_latest_checkpoint() -> Path:
    """Return the checkpoint with the highest step number."""
    ckpts = sorted(CHECKPOINT_DIR.glob("sae_step_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {CHECKPOINT_DIR}")
    return ckpts[-1]


def _load_sae(ckpt_path: Path, device: torch.device) -> SparseAutoencoder:
    """Load an SAE from a checkpoint file."""
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    d_model = ckpt["d_model"]
    dict_size = ckpt["dict_size"]
    sae = SparseAutoencoder(d_model, dict_size, DECODER_INIT_NORM)
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.to(device).eval()
    return sae


def _load_activations() -> np.ndarray:
    """Load the pre-extracted activations (memmap)."""
    meta_path = ACTIVATIONS_DIR / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    act_path = ACTIVATIONS_DIR / meta["act_file"]
    return np.load(str(act_path), mmap_mode="r")


# ---------------------------------------------------------------------------
# Post-training normalisation
# ---------------------------------------------------------------------------

def normalise_decoder(sae: SparseAutoencoder):
    """Rescale weights so decoder columns have unit L2 norm.

    After this transformation the model is mathematically equivalent, but
    feature activations are now in "true" units and decoder vectors are
    unit directions.

        W_enc'[i] = W_enc[i]  * ||W_dec[:, i]||
        b_enc'[i] = b_enc[i]  * ||W_dec[:, i]||
        W_dec'[:, i] = W_dec[:, i] / ||W_dec[:, i]||
        b_dec  unchanged
    """
    with torch.no_grad():
        norms = sae.W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)  # (1, F)
        sae.W_enc.mul_(norms.T)       # (F, d) * (F, 1)
        sae.b_enc.mul_(norms.squeeze(0))  # (F,)
        sae.W_dec.div_(norms)          # (d, F) / (1, F)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{GPU_IDS[0]}" if torch.cuda.is_available()
                          else "cpu")

    # ---- Load checkpoint & activations ------------------------------------
    ckpt_path = _find_latest_checkpoint()
    print(f"[analyse] Loading checkpoint: {ckpt_path.name}")
    sae = _load_sae(ckpt_path, device)

    print("[analyse] Normalising decoder to unit-norm columns ...")
    normalise_decoder(sae)

    acts_mmap = _load_activations()
    N_total = acts_mmap.shape[0]
    d_model = acts_mmap.shape[1]
    dict_size = sae.dict_size
    n_tokens = min(ANALYSIS_NUM_TOKENS, N_total)
    print(f"[analyse] Analysing {n_tokens:,} / {N_total:,} tokens  "
          f"(d={d_model}, F={dict_size})")

    # ---- Accumulate statistics --------------------------------------------
    batch_size = 4096
    n_batches = (n_tokens + batch_size - 1) // batch_size

    # Running accumulators.
    feature_fire_count = torch.zeros(dict_size, device=device)   # how often each feature > 0
    feature_max_act = torch.zeros(dict_size, device=device)      # max activation per feature
    total_l0 = 0.0
    total_mse = 0.0
    total_var = 0.0
    tokens_seen = 0

    # For top-k tracking on a sample of features.
    sample_feature_ids = np.linspace(0, dict_size - 1,
                                     min(ANALYSIS_SAMPLE_FEATURES, dict_size),
                                     dtype=int)
    sample_feature_ids = np.unique(sample_feature_ids)
    # Store (activation_value, global_token_index) per sampled feature.
    topk_heap: dict[int, list[tuple[float, int]]] = {
        int(fid): [] for fid in sample_feature_ids
    }

    global_idx = 0
    for batch_i in tqdm(range(n_batches), desc="analysing"):
        start = batch_i * batch_size
        end = min(start + batch_size, n_tokens)
        x_np = acts_mmap[start:end]
        x = torch.from_numpy(np.array(x_np)).float().to(device)

        with torch.no_grad():
            f = sae.encode(x)
            x_hat = sae.decode(f)

        # MSE & variance components.
        total_mse += (x - x_hat).pow(2).sum().item()
        total_var += (x - x.mean(dim=0, keepdim=True)).pow(2).sum().item()

        # L0.
        active_mask = f > 0
        total_l0 += active_mask.float().sum(dim=-1).sum().item()  # sum across batch

        # Per-feature stats.
        feature_fire_count += active_mask.float().sum(dim=0)
        feature_max_act = torch.max(feature_max_act, f.max(dim=0).values)

        tokens_seen += (end - start)

        # Top-k for sampled features.
        f_sample = f[:, sample_feature_ids].cpu().numpy()  # (B, n_sample)
        for j, fid in enumerate(sample_feature_ids):
            fid = int(fid)
            col = f_sample[:, j]
            for local_i in range(col.shape[0]):
                val = float(col[local_i])
                if val <= 0:
                    continue
                tok_idx = global_idx + local_i
                heap = topk_heap[fid]
                if len(heap) < ANALYSIS_TOP_K:
                    heap.append((val, tok_idx))
                elif val > heap[-1][0]:
                    heap[-1] = (val, tok_idx)
                # Keep sorted descending.
                heap.sort(key=lambda t: -t[0])

        global_idx += (end - start)

    # ---- Aggregate --------------------------------------------------------
    mean_l0 = total_l0 / tokens_seen
    explained_var = 1.0 - total_mse / max(total_var, 1e-12)

    feature_freq = (feature_fire_count / tokens_seen).cpu().numpy()
    dead_mask = feature_fire_count == 0
    n_dead = int(dead_mask.sum().item())
    n_alive = dict_size - n_dead
    pct_dead = 100.0 * n_dead / dict_size

    feature_max_act_np = feature_max_act.cpu().numpy()

    # ---- Decoder cosine similarity for sampled features -------------------
    print("[analyse] Computing decoder neighborhoods ...")
    W_dec_normed = sae.W_dec.detach()  # already unit-norm after normalise_decoder
    neighborhoods: dict[str, list[dict]] = {}
    for fid in sample_feature_ids:
        fid = int(fid)
        col = W_dec_normed[:, fid]  # (d,)
        cos = (W_dec_normed.T @ col)  # (F,)
        cos[fid] = -2.0  # exclude self
        topk_vals, topk_ids = cos.topk(5)
        neighbors = [
            {"feature": int(topk_ids[k]), "cosine": float(topk_vals[k])}
            for k in range(5)
        ]
        neighborhoods[str(fid)] = neighbors

    # ---- Build report -----------------------------------------------------
    report = {
        "checkpoint": ckpt_path.name,
        "tokens_analysed": tokens_seen,
        "d_model": d_model,
        "dict_size": dict_size,
        "mean_l0": round(mean_l0, 2),
        "explained_variance": round(explained_var, 6),
        "dead_features": n_dead,
        "alive_features": n_alive,
        "pct_dead": round(pct_dead, 2),
        "feature_freq_percentiles": {
            p: round(float(np.percentile(feature_freq, p)), 8)
            for p in [0, 10, 25, 50, 75, 90, 99, 100]
        },
        "feature_max_act_percentiles": {
            p: round(float(np.percentile(feature_max_act_np, p)), 4)
            for p in [0, 10, 25, 50, 75, 90, 99, 100]
        },
        "sample_feature_topk": {
            str(fid): [
                {"activation": round(v, 4), "token_index": idx}
                for v, idx in topk_heap[fid]
            ]
            for fid in [int(x) for x in sample_feature_ids]
        },
        "decoder_neighborhoods": neighborhoods,
    }

    # ---- Save & print -----------------------------------------------------
    report_path = ANALYSIS_DIR / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n[analyse] Report saved to {report_path}")

    print("\n===== SAE Analysis Summary =====")
    print(f"  Checkpoint        : {ckpt_path.name}")
    print(f"  Tokens analysed   : {tokens_seen:,}")
    print(f"  d_model           : {d_model}")
    print(f"  dict_size         : {dict_size:,}")
    print(f"  Mean L0           : {mean_l0:.2f}")
    print(f"  Explained variance: {explained_var:.4f}")
    print(f"  Dead features     : {n_dead:,} / {dict_size:,} ({pct_dead:.1f}%)")
    print(f"  Alive features    : {n_alive:,}")
    print("================================\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    analyse()


if __name__ == "__main__":
    main()
