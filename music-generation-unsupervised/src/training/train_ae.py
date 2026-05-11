"""Training entrypoint for Task 1 (LSTM Autoencoder)."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.config import PIANO_ROLL, TASK1
from src.models.autoencoder import AutoencoderHParams, LSTMAutoencoder


class PianoRollWindowsDataset(Dataset):
    def __init__(self, npy_path: Path):
        self.npy_path = npy_path
        # Memory-map to avoid loading everything into RAM.
        self.data = np.load(npy_path, mmap_mode="r")

        if self.data.ndim != 3 or self.data.shape[1:] != (PIANO_ROLL.window_steps, 88):
            raise ValueError(f"Unexpected dataset shape {self.data.shape} in {npy_path}")

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = self.data[idx].astype(np.float32)  # (T, 88)
        return torch.from_numpy(x)


def binary_focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma: float = 2.0,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Binary focal loss using logits.

    logits/targets shape: (B, T, 88). targets in {0,1}.
    """

    # BCE with logits (optionally class-weighted for positives)
    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight,
    )
    probs = torch.sigmoid(logits)
    pt = torch.where(targets >= 0.5, probs, 1.0 - probs)
    focal = (1.0 - pt).pow(gamma)
    loss = focal * bce
    return loss.mean()


def estimate_pos_weight_from_dataset(
    dataset: Dataset,
    *,
    max_batches: int = 200,
    batch_size: int = 64,
    seed: int = 42,
) -> float:
    """Estimate negative/positive ratio from a subset for stability.

    We clamp to the recommended range [10, 30] (supplementary guide).
    """

    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)
    pos = 0.0
    total = 0.0
    for i, x in enumerate(loader):
        if i >= max_batches:
            break
        pos += float(x.sum().item())
        total += float(x.numel())

    if pos <= 0.0:
        return 1.0
    neg = total - pos
    ratio = neg / pos
    return float(max(10.0, min(30.0, ratio)))


def run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    gamma: float,
    pos_weight: Optional[torch.Tensor],
    grad_clip: float,
    log_every: int = 0,
    label: str = "",
) -> float:
    is_train = optimizer is not None
    model.train(is_train)

    losses = []
    total_batches = len(loader)
    for batch_idx, x in enumerate(loader, start=1):
        x = x.to(device)
        logits = model(x)
        loss = binary_focal_loss_with_logits(logits, x, gamma=gamma, pos_weight=pos_weight)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        losses.append(float(loss.detach().cpu().item()))

        if log_every and (batch_idx % log_every == 0 or batch_idx == total_batches):
            avg = float(np.mean(losses[-min(len(losses), 20) :]))
            prefix = (label + " ") if label else ""
            print(f"{prefix}batch {batch_idx:>4}/{total_batches} | loss~{avg:.4f}", flush=True)

    return float(np.mean(losses)) if losses else float("nan")


def train_task1(
    *,
    train_npy: Path,
    val_npy: Path,
    checkpoint_path: Path,
    loss_curve_path: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_dim: int,
    latent_dim: int,
    num_layers: int,
    dropout: float,
    gamma: float,
    pos_weight: Optional[float],
    grad_clip: float,
    seed: int,
    device: str,
    patience: int,
    min_delta: float,
    use_scheduler: bool,
    log_every: int,
) -> Tuple[list[float], list[float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    dev = torch.device(device)

    train_ds = PianoRollWindowsDataset(train_npy)
    val_ds = PianoRollWindowsDataset(val_npy)

    if pos_weight is None:
        pos_weight = estimate_pos_weight_from_dataset(train_ds, batch_size=batch_size, seed=seed)

    print(f"Using pos_weight={pos_weight:.2f} gamma={gamma:.2f}")

    # pos_weight in BCEWithLogitsLoss expects a tensor; broadcast across all cells
    pos_weight_t = torch.tensor([pos_weight], device=dev)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    hparams = AutoencoderHParams(
        input_dim=88,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    model = LSTMAutoencoder(hparams).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )

    train_losses: list[float] = []
    val_losses: list[float] = []

    best_val = float("inf")
    epochs_since_improve = 0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    loss_curve_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        tr = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=dev,
            gamma=gamma,
            pos_weight=pos_weight_t,
            grad_clip=grad_clip,
            log_every=log_every,
            label=f"epoch {epoch:03d} train",
        )
        va = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=dev,
            gamma=gamma,
            pos_weight=pos_weight_t,
            grad_clip=0.0,
            log_every=log_every,
            label=f"epoch {epoch:03d} val",
        )

        train_losses.append(tr)
        val_losses.append(va)

        if va < best_val:
            best_val = va
            epochs_since_improve = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "hparams": asdict(hparams),
                    "fs": PIANO_ROLL.fs,
                    "pitch_min": PIANO_ROLL.pitch_min,
                    "pitch_max": PIANO_ROLL.pitch_max,
                    "velocity": PIANO_ROLL.default_velocity,
                    "window_steps": PIANO_ROLL.window_steps,
                    "min_active_fraction": PIANO_ROLL.min_active_fraction,
                    "train_loss": tr,
                    "val_loss": va,
                    "epoch": epoch,
                    "pos_weight": float(pos_weight),
                    "gamma": float(gamma),
                },
                checkpoint_path,
            )
        else:
            epochs_since_improve += 1

        if scheduler is not None:
            scheduler.step(va)

        print(
            f"Epoch {epoch:03d}/{epochs} | train={tr:.4f} val={va:.4f} | best_val={best_val:.4f}"
        )

        if patience > 0 and epochs_since_improve >= patience:
            print(f"Early stopping: no val improvement for {patience} epochs")
            break

    # Plot loss curve (deliverable)
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction loss (focal BCE)")
    plt.title("Task 1: LSTM Autoencoder Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_curve_path, dpi=200)
    plt.close()

    return train_losses, val_losses


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Task 1 LSTM Autoencoder")
    p.add_argument("--train_npy", type=Path, default=TASK1.train_npy)
    p.add_argument("--val_npy", type=Path, default=TASK1.val_npy)
    p.add_argument("--checkpoint", type=Path, default=TASK1.checkpoint_path)
    p.add_argument("--loss_curve", type=Path, default=TASK1.loss_curve_path)

    # CPU-friendly defaults (you can increase hidden_dim/epochs if you have a GPU)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)

    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument(
        "--pos_weight",
        type=float,
        default=None,
        help="If omitted, estimated from training data and clamped to [10,30]",
    )
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--min_delta", type=float, default=0.0)
    p.add_argument("--use_scheduler", action="store_true")
    p.add_argument(
        "--log_every",
        type=int,
        default=0,
        help="If >0, prints batch progress every N batches (useful on CPU)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    train_task1(
        train_npy=args.train_npy,
        val_npy=args.val_npy,
        checkpoint_path=args.checkpoint,
        loss_curve_path=args.loss_curve,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        gamma=args.gamma,
        pos_weight=args.pos_weight,
        grad_clip=args.grad_clip,
        seed=args.seed,
        device=args.device,
        patience=args.patience,
        min_delta=args.min_delta,
        use_scheduler=bool(args.use_scheduler),
        log_every=int(args.log_every),
    )


if __name__ == "__main__":
    main()
