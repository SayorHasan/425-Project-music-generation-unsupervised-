"""Training entrypoint for Task 2 (VAE)."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.config import PIANO_ROLL, TASK2
from src.models.vae import LSTMVAE, VAEHParams


class PianoRollWindowsDataset(Dataset):
    def __init__(self, npy_path: Path):
        self.npy_path = npy_path
        self.data = np.load(npy_path, mmap_mode="r")
        if self.data.ndim != 3 or self.data.shape[1:] != (PIANO_ROLL.window_steps, 88):
            raise ValueError(f"Unexpected dataset shape {self.data.shape} in {npy_path}")

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.data[idx].astype(np.float32))


def vae_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    kl_beta: float,
    pos_weight: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean", pos_weight=pos_weight)
    kl_per_sample = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl = kl_per_sample.mean()
    total = recon + kl_beta * kl
    return total, recon, kl


def estimate_pos_weight_from_dataset(dataset: Dataset, *, max_batches: int = 200, batch_size: int = 64, seed: int = 42) -> float:
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
    ratio = (total - pos) / pos
    return float(max(10.0, min(30.0, ratio)))


def run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    kl_beta: float,
    pos_weight: Optional[torch.Tensor],
    grad_clip: float,
    log_every: int = 0,
    label: str = "",
) -> tuple[float, float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_losses = []
    recon_losses = []
    kl_losses = []
    total_batches = len(loader)
    for batch_idx, x in enumerate(loader, start=1):
        x = x.to(device)
        logits, mu, logvar = model(x)
        total, recon, kl = vae_loss(logits, x, mu, logvar, kl_beta=kl_beta, pos_weight=pos_weight)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
        total_losses.append(float(total.detach().cpu().item()))
        recon_losses.append(float(recon.detach().cpu().item()))
        kl_losses.append(float(kl.detach().cpu().item()))
        if log_every and (batch_idx % log_every == 0 or batch_idx == total_batches):
            prefix = (label + " ") if label else ""
            print(f"{prefix}batch {batch_idx:>4}/{total_batches} | total~{np.mean(total_losses[-20:]):.4f} recon~{np.mean(recon_losses[-20:]):.4f} kl~{np.mean(kl_losses[-20:]):.4f}", flush=True)
    return float(np.mean(total_losses)), float(np.mean(recon_losses)), float(np.mean(kl_losses))


def train_task2(
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
    kl_beta: float,
    kl_warmup_epochs: int,
    pos_weight: Optional[float],
    grad_clip: float,
    seed: int,
    device: str,
    patience: int,
    use_scheduler: bool,
    log_every: int,
) -> tuple[list[float], list[float], list[float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    train_ds = PianoRollWindowsDataset(train_npy)
    val_ds = PianoRollWindowsDataset(val_npy)
    if pos_weight is None:
        pos_weight = estimate_pos_weight_from_dataset(train_ds, batch_size=batch_size, seed=seed)
    print(f"Using pos_weight={pos_weight:.2f} kl_beta={kl_beta:.2f}")
    pos_weight_t = torch.tensor([pos_weight], device=dev)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    hparams = VAEHParams(input_dim=88, hidden_dim=hidden_dim, latent_dim=latent_dim, num_layers=num_layers, dropout=dropout)
    model = LSTMVAE(hparams).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    train_total: list[float] = []
    val_total: list[float] = []
    val_recon: list[float] = []
    val_kl: list[float] = []
    best_val = float("inf")
    epochs_since_improve = 0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    loss_curve_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        if kl_warmup_epochs > 0:
            effective_kl_beta = kl_beta * min(1.0, epoch / float(kl_warmup_epochs))
        else:
            effective_kl_beta = kl_beta

        tr_total, tr_recon, tr_kl = run_epoch(model=model, loader=train_loader, optimizer=optimizer, device=dev, kl_beta=effective_kl_beta, pos_weight=pos_weight_t, grad_clip=grad_clip, log_every=log_every, label=f"epoch {epoch:03d} train")
        va_total, va_recon, va_kl = run_epoch(model=model, loader=val_loader, optimizer=None, device=dev, kl_beta=effective_kl_beta, pos_weight=pos_weight_t, grad_clip=0.0, log_every=log_every, label=f"epoch {epoch:03d} val")

        train_total.append(tr_total)
        val_total.append(va_total)
        val_recon.append(va_recon)
        val_kl.append(va_kl)

        if va_total < best_val:
            best_val = va_total
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
                    "kl_beta": float(kl_beta),
                    "kl_warmup_epochs": int(kl_warmup_epochs),
                    "pos_weight": float(pos_weight),
                    "epoch": epoch,
                },
                checkpoint_path,
            )
        else:
            epochs_since_improve += 1

        if scheduler is not None:
            scheduler.step(va_total)

        print(f"Epoch {epoch:03d}/{epochs} | beta={effective_kl_beta:.3f} train={tr_total:.4f} val={va_total:.4f} recon={va_recon:.4f} kl={va_kl:.4f} | best_val={best_val:.4f}")
        if patience > 0 and epochs_since_improve >= patience:
            print(f"Early stopping: no val improvement for {patience} epochs")
            break

    plt.figure(figsize=(8, 4))
    plt.plot(train_total, label="train total")
    plt.plot(val_total, label="validation total")
    plt.plot(val_recon, label="validation recon")
    plt.plot(val_kl, label="validation KL")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Task 2: VAE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_curve_path, dpi=200)
    plt.close()

    return train_total, val_total, val_kl


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Task 2 LSTM VAE")
    p.add_argument("--train_npy", type=Path, default=TASK2.train_npy)
    p.add_argument("--val_npy", type=Path, default=TASK2.val_npy)
    p.add_argument("--checkpoint", type=Path, default=TASK2.checkpoint_path)
    p.add_argument("--loss_curve", type=Path, default=TASK2.loss_curve_path)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--kl_beta", type=float, default=0.1)
    p.add_argument("--kl_warmup_epochs", type=int, default=10)
    p.add_argument("--pos_weight", type=float, default=None)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--use_scheduler", action="store_true")
    p.add_argument("--log_every", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    train_task2(
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
        kl_beta=args.kl_beta,
        kl_warmup_epochs=args.kl_warmup_epochs,
        pos_weight=args.pos_weight,
        grad_clip=args.grad_clip,
        seed=args.seed,
        device=args.device,
        patience=args.patience,
        use_scheduler=bool(args.use_scheduler),
        log_every=int(args.log_every),
    )


if __name__ == "__main__":
    main()
