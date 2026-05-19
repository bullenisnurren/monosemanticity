#!/usr/bin/env python3
"""
Train a Sparse Autoencoder (SAE) on pre-extracted activations.

Architecture (from "Scaling Monosemanticity"):
    encode:  f = ReLU( W_enc @ (x - b_dec) + b_enc )
    decode:  x_hat = W_dec @ f + b_dec
    loss:    MSE(x, x_hat)  +  lambda * sum_i  f_i * ||w_dec_i||_2

Key training details:
  - Adam (no weight decay),  betas from constants.
  - L1 coefficient linearly warmed up over first 5 % of steps.
  - Learning-rate linearly decayed to 0 over last 20 % of steps.
  - Gradient-norm clipped to 1.0.
  - Decoder columns are NOT constrained to unit norm during training.
    Instead, the L1 term is weighted by the decoder column norms.
"""

import json
import warnings
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from constants import (
    EXPANSION_FACTOR,
    BATCH_SIZE,
    BUFFER_SEQUENCES,
    NUM_TRAINING_STEPS,
    L1_COEFF,
    L1_WARMUP_FRAC,
    LR,
    LR_DECAY_FRAC,
    ADAM_BETA1,
    ADAM_BETA2,
    GRAD_CLIP_NORM,
    DECODER_INIT_NORM,
    NUM_GPUS,
    GPU_IDS,
    ACTIVATIONS_TRAIN_DIR,
    CHECKPOINT_DIR,
    CHECKPOINT_EVERY,
    LOG_EVERY,
)


# ---------------------------------------------------------------------------
# SAE model
# ---------------------------------------------------------------------------

class SparseAutoencoder(nn.Module):
    """Sparse autoencoder following the Anthropic "Scaling Mono." recipe."""

    def __init__(self, d_model: int, dict_size: int, decoder_init_norm: float):
        super().__init__()
        self.d_model = d_model
        self.dict_size = dict_size

        # Decoder: W_dec @ f + b_dec  ---  shape (d_model, dict_size)
        self.W_dec = nn.Parameter(torch.empty(d_model, dict_size))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        # Encoder: ReLU(W_enc @ (x - b_dec) + b_enc)  ---  shape (dict_size, d_model)
        self.W_enc = nn.Parameter(torch.empty(dict_size, d_model))
        self.b_enc = nn.Parameter(torch.zeros(dict_size))

        self._init_weights(decoder_init_norm)

    # ---- initialisation ---------------------------------------------------

    def _init_weights(self, decoder_init_norm: float):
        """Paper prescription: random directions with fixed norm, W_enc = W_dec^T."""
        # Random unit directions.
        nn.init.normal_(self.W_dec)
        with torch.no_grad():
            norms = self.W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
            self.W_dec.mul_(decoder_init_norm / norms)
            # Encoder initialised as transpose of decoder.
            self.W_enc.copy_(self.W_dec.T)

    # ---- forward ----------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations to feature coefficients."""
        x_centred = x - self.b_dec
        return torch.relu(x_centred @ self.W_enc.T + self.b_enc)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """Decode feature coefficients back to activation space."""
        return f @ self.W_dec.T + self.b_dec

    def forward(self, x: torch.Tensor, l1_coeff: float = 0.0):
        """
        Parameters
        ----------
        x        : (batch, d_model)
        l1_coeff : current L1 coefficient (pass 0 to skip loss computation)

        Returns
        -------
        loss     : scalar  (mean loss across the batch)
        mse_loss : scalar  (detached, for logging)
        l1_loss  : scalar  (detached, for logging; before coeff multiplier)
        l0       : scalar  (detached, mean number of active features)
        """
        f = self.encode(x)            # (B, F)
        x_hat = self.decode(f)        # (B, d)
        dec_norms = self.W_dec.norm(dim=0)  # (F,)

        mse_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        # Weighted L1: sum_i f_i * ||w_dec_i||  (per sample, then mean).
        l1_loss = (f * dec_norms.unsqueeze(0)).sum(dim=-1).mean()
        loss = mse_loss + l1_coeff * l1_loss

        with torch.no_grad():
            l0 = (f > 0).float().sum(dim=-1).mean()

        return loss, mse_loss.detach(), l1_loss.detach(), l0.detach()


# ---------------------------------------------------------------------------
# Activation dataloader (reads from memmap)
# ---------------------------------------------------------------------------

class ActivationLoader:
    """Iterate over pre-extracted activations with HDD-friendly access.

    On-disk shape: ``(num_sequences, SEQ_LEN, d_model)``, raw / unnormalised.

    Naive token-level shuffling triggers ``BATCH_SIZE`` random seeks per batch
    into a multi-GB memmap, which is fatal on spinning disks.  Instead we keep
    a RAM-resident buffer of ``buffer_sequences`` whole sequences (read with
    sorted IDs so the disk head moves monotonically), shuffle the *tokens*
    inside the buffer once, and drain it batch-by-batch.  The next buffer is
    prefetched on a worker thread so I/O overlaps with GPU compute.

    Quality argument: with ``buffer_sequences`` ≫ ``batch_size``, each batch
    of ``batch_size`` tokens is sampled from ``buffer_sequences`` distinct
    documents (so within-batch context correlation is negligible).  This is
    indistinguishable in practice from full token-level shuffling for SAE
    training, while avoiding the random-seek penalty.
    """

    def __init__(self, activations_dir: Path, batch_size: int,
                 buffer_sequences: int):
        meta_path = activations_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No meta.json in {activations_dir}. Run extract.py first."
            )
        with open(meta_path) as f:
            self.meta = json.load(f)

        act_path = activations_dir / self.meta["act_file"]
        # Memmapped (n_seqs, seq_len, d_model) — reads on slicing.
        self._acts3d = np.load(str(act_path), mmap_mode="r")
        self.n_seqs, self.seq_len, self.d = self._acts3d.shape
        self.N = self.n_seqs * self.seq_len
        self.batch_size = batch_size
        self.scale = float(self.meta["scale"])

        if buffer_sequences > self.n_seqs:
            buffer_sequences = self.n_seqs
        if buffer_sequences * self.seq_len < batch_size:
            raise ValueError(
                f"buffer_sequences*seq_len ({buffer_sequences * self.seq_len}) "
                f"< batch_size ({batch_size})"
            )
        self.buffer_sequences = buffer_sequences
        self.buffer_tokens = buffer_sequences * self.seq_len

        # Sequence-level permutation reused across buffer fills.
        self._seq_perm = np.random.permutation(self.n_seqs).astype(np.int64)
        self._seq_ptr = 0

        # Background prefetch.
        self._exec = ThreadPoolExecutor(max_workers=1,
                                         thread_name_prefix="actloader")
        self._buffer: np.ndarray | None = None
        self._buffer_ptr = 0
        self._next_future = self._exec.submit(self._make_buffer)

    # ------------------------------------------------------------------
    def _next_seq_block(self) -> np.ndarray:
        """Return the next ``buffer_sequences`` sorted sequence indices."""
        if self._seq_ptr + self.buffer_sequences > self.n_seqs:
            self._seq_perm = np.random.permutation(self.n_seqs).astype(np.int64)
            self._seq_ptr = 0
        ids = self._seq_perm[self._seq_ptr:self._seq_ptr + self.buffer_sequences]
        self._seq_ptr += self.buffer_sequences
        # Sorting → monotonic disk head movement during the read pass.
        return np.sort(ids)

    def _make_buffer(self) -> np.ndarray:
        """Read a fresh shuffled buffer (called on the background thread)."""
        ids = self._next_seq_block()
        buf = np.empty((self.buffer_sequences, self.seq_len, self.d),
                        dtype=np.float32)
        for i, sid in enumerate(ids):
            # Each indexing is a contiguous (seq_len*d_model*4)-byte read.
            buf[i] = self._acts3d[int(sid)]
        flat = buf.reshape(-1, self.d)
        # Token-level shuffle inside the loaded buffer.
        np.random.shuffle(flat)
        # Apply global normalisation up-front so get_batch is allocation-free.
        flat *= self.scale
        return flat

    # ------------------------------------------------------------------
    def get_batch(self, device: torch.device) -> torch.Tensor:
        """Return a normalised ``(batch_size, d)`` float32 tensor on *device*."""
        if (self._buffer is None
                or self._buffer_ptr + self.batch_size > len(self._buffer)):
            self._buffer = self._next_future.result()
            self._buffer_ptr = 0
            self._next_future = self._exec.submit(self._make_buffer)
        start = self._buffer_ptr
        end = start + self.batch_size
        self._buffer_ptr = end
        # from_numpy shares memory; .to(device) copies to GPU, after which the
        # buffer slice can be safely overwritten on the next refill.
        return torch.from_numpy(self._buffer[start:end]).to(device,
                                                              non_blocking=True)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def _lr_lambda(step: int, total_steps: int, decay_frac: float) -> float:
    """Constant LR with linear decay over the last *decay_frac* of training."""
    decay_start = int(total_steps * (1.0 - decay_frac))
    if step < decay_start:
        return 1.0
    # Linear decay from 1 -> 0.
    return max(0.0, 1.0 - (step - decay_start) / (total_steps - decay_start))


# ---------------------------------------------------------------------------
# Checkpoint discovery / resume
# ---------------------------------------------------------------------------

def _find_latest_checkpoint() -> Path | None:
    """Return the highest-numbered checkpoint in CHECKPOINT_DIR, or None."""
    ckpts = sorted(CHECKPOINT_DIR.glob("sae_step_*.pt"))
    return ckpts[-1] if ckpts else None


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Activation loader ------------------------------------------------
    loader = ActivationLoader(ACTIVATIONS_TRAIN_DIR, BATCH_SIZE, BUFFER_SEQUENCES)
    d_model = loader.d
    dict_size = d_model * EXPANSION_FACTOR
    print(f"[train] Activation scale = {loader.scale:.6f}")
    print(f"[train] Loader buffer: {loader.buffer_sequences:,} sequences "
          f"({loader.buffer_tokens:,} tokens, "
          f"{loader.buffer_sequences * loader.seq_len * loader.d * 4 / 1e9:.2f} GB)")
    print(f"[train] d_model={d_model}, dict_size={dict_size} "
          f"({EXPANSION_FACTOR}x expansion)")
    print(f"[train] {loader.N:,} activation vectors, batch_size={BATCH_SIZE}")
    print(f"[train] Training for {NUM_TRAINING_STEPS:,} steps")

    steps_per_epoch = loader.N // BATCH_SIZE
    if steps_per_epoch > 0:
        epochs = NUM_TRAINING_STEPS / steps_per_epoch
        print(f"[train] Steps per epoch: {steps_per_epoch:,}  ->  "
              f"~{epochs:.2f} epoch(s)")

    # ---- Model ------------------------------------------------------------
    primary_device = torch.device(f"cuda:{GPU_IDS[0]}")
    sae = SparseAutoencoder(d_model, dict_size, DECODER_INIT_NORM).to(primary_device)
    param_count = sum(p.numel() for p in sae.parameters())
    print(f"[train] SAE parameters: {param_count:,} "
          f"({param_count * 4 / 1e9:.2f} GB @ fp32)")

    # Wrap in DataParallel when using multiple GPUs.
    model = sae
    if NUM_GPUS > 1:
        model = nn.DataParallel(sae, device_ids=GPU_IDS)
        # DataParallel gathers scalar outputs by unsqueezing → harmless warning.
        warnings.filterwarnings(
            "ignore", message="Was asked to gather along dimension 0"
        )
        print(f"[train] Using DataParallel across GPUs {GPU_IDS}")

    # ---- Optimiser --------------------------------------------------------
    optimiser = torch.optim.Adam(
        sae.parameters(),
        lr=LR,
        betas=(ADAM_BETA1, ADAM_BETA2),
        weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser,
        lr_lambda=lambda step: _lr_lambda(step, NUM_TRAINING_STEPS, LR_DECAY_FRAC),
    )

    # ---- Resume from latest checkpoint (if any) --------------------------
    start_step = 0
    latest_ckpt = _find_latest_checkpoint()
    if latest_ckpt is not None:
        print(f"[train] Resuming from {latest_ckpt.name}")
        ckpt = torch.load(str(latest_ckpt), map_location=primary_device,
                          weights_only=False)
        if ckpt["d_model"] != d_model or ckpt["dict_size"] != dict_size:
            raise RuntimeError(
                f"Checkpoint shape mismatch: ckpt has "
                f"d_model={ckpt['d_model']}, dict_size={ckpt['dict_size']}; "
                f"loader has d_model={d_model}, dict_size={dict_size}.  "
                f"Move/clear {CHECKPOINT_DIR} or fix the config."
            )
        sae.load_state_dict(ckpt["model_state_dict"])
        optimiser.load_state_dict(ckpt["optimiser_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = int(ckpt["step"])
        if start_step >= NUM_TRAINING_STEPS:
            print(f"[train] Latest checkpoint is already at step "
                  f"{start_step:,} >= NUM_TRAINING_STEPS={NUM_TRAINING_STEPS:,}.  "
                  f"Nothing to do.")
            return

    # ---- L1 warmup --------------------------------------------------------
    l1_warmup_steps = int(NUM_TRAINING_STEPS * L1_WARMUP_FRAC)

    def _current_l1(step: int) -> float:
        if l1_warmup_steps == 0:
            return L1_COEFF
        return L1_COEFF * min(1.0, step / l1_warmup_steps)

    # ---- Training ---------------------------------------------------------
    log_mse = 0.0
    log_l1 = 0.0
    log_loss = 0.0
    log_l0 = 0.0
    t0 = time.time()

    pbar = tqdm(range(start_step + 1, NUM_TRAINING_STEPS + 1),
                desc="training", unit="step",
                initial=start_step, total=NUM_TRAINING_STEPS)
    for step in pbar:
        x = loader.get_batch(primary_device)  # (B, d)

        l1_coeff = _current_l1(step)
        # Forward pass.  DataParallel splits x across GPUs; each replica
        # computes its own per-shard loss.  DP gathers the scalar outputs
        # into a 1-D tensor (one element per GPU), so we .mean() before
        # calling backward.
        loss, mse, l1, l0 = model(x, l1_coeff)
        loss = loss.mean()

        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), GRAD_CLIP_NORM)
        optimiser.step()
        scheduler.step()

        # ---- Logging bookkeeping -----------------------------------------
        log_loss += loss.item()
        log_mse += mse.mean().item()
        log_l1 += l1.mean().item()
        log_l0 += l0.mean().item()

        if step % LOG_EVERY == 0:
            n = LOG_EVERY
            # Explained variance over last log window (approximated from
            # running averages -- exact EV is computed in analyse.py).
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix_str(
                f"loss={log_loss/n:.4f}  mse={log_mse/n:.4f}  "
                f"l1={log_l1/n:.4f}  L0={log_l0/n:.1f}  "
                f"l1c={l1_coeff:.3f}  lr={current_lr:.2e}"
            )
            log_loss = log_mse = log_l1 = log_l0 = 0.0

        # ---- Checkpoint ---------------------------------------------------
        if step % CHECKPOINT_EVERY == 0 or step == NUM_TRAINING_STEPS:
            _save_checkpoint(sae, optimiser, scheduler, step,
                             d_model, dict_size, loader.scale)

    elapsed = time.time() - t0
    print(f"[train] Finished {NUM_TRAINING_STEPS} steps in {elapsed:.1f}s "
          f"({elapsed / NUM_TRAINING_STEPS:.3f}s/step)")


def _save_checkpoint(sae, optimiser, scheduler, step,
                     d_model, dict_size, activation_scale):
    path = CHECKPOINT_DIR / f"sae_step_{step:06d}.pt"
    torch.save(
        {
            "step": step,
            "d_model": d_model,
            "dict_size": dict_size,
            "expansion_factor": EXPANSION_FACTOR,
            "activation_scale": float(activation_scale),
            "model_state_dict": sae.state_dict(),
            "optimiser_state_dict": optimiser.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": {
                "l1_coeff": L1_COEFF,
                "lr": LR,
                "batch_size": BATCH_SIZE,
                "num_training_steps": NUM_TRAINING_STEPS,
                "decoder_init_norm": DECODER_INIT_NORM,
            },
        },
        str(path),
    )
    print(f"  [checkpoint] saved -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    train()


if __name__ == "__main__":
    main()
