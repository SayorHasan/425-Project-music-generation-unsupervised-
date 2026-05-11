"""Training entrypoint for Task 3 (Transformer)."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from evaluation.metrics import evaluate_midi_file, summarize_generated_directory
from evaluation.pitch_histogram import aggregate_pitch_class_histogram
from generation.midi_export import validate_midi
from src.config import MAESTRO_CSV, MAESTRO_ROOT, TASK3
from src.models.transformer import DecoderOnlyTransformer, TransformerHParams
from src.preprocessing.tokenizer import (
    Task3TokenConfig,
    Task3TokenDataset,
    build_task3_token_dataset,
    load_maestro_split,
    load_task3_metadata,
    load_task3_tokenizer,
    task3_collate_batch,
)


def _ensure_task3_data(
    *,
    maestro_root: Path,
    maestro_csv: Path,
    limit_files: Optional[int],
    rebuild: bool,
    chunk_length: int,
    stride: int,
) -> dict[str, object]:
    if rebuild or not TASK3.train_jsonl.exists() or not TASK3.val_jsonl.exists():
        return build_task3_token_dataset(
            maestro_root=maestro_root,
            maestro_csv=maestro_csv,
            out_dir=TASK3.token_dir,
            config=Task3TokenConfig(chunk_length=chunk_length, stride=stride),
            limit_files=limit_files,
        )
    return load_task3_metadata(TASK3.metadata_json)


def _prepare_loader(dataset: Task3TokenDataset, *, batch_size: int, shuffle: bool, pad_id: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: task3_collate_batch(batch, pad_id=pad_id),
    )


def _shift_batch(token_batch: torch.Tensor, pad_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = token_batch[:, :-1].contiguous()
    targets = token_batch[:, 1:].contiguous()
    attention_mask = inputs != pad_id
    return inputs, targets, attention_mask


def _cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> torch.Tensor:
    return F.cross_entropy(
        logits.view(-1, logits.shape[-1]),
        targets.view(-1),
        ignore_index=pad_id,
    )


def _run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    pad_id: int,
    log_every: int = 0,
    label: str = "",
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    losses: list[float] = []
    total_batches = len(loader)
    for batch_idx, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device)
        genre_ids = batch["genre_ids"].to(device)
        inputs, targets, attention_mask = _shift_batch(input_ids, pad_id)
        logits = model(inputs, genre_ids=genre_ids, attention_mask=attention_mask)
        loss = _cross_entropy_loss(logits, targets, pad_id)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        if log_every and (batch_idx % log_every == 0 or batch_idx == total_batches):
            prefix = f"{label} " if label else ""
            print(f"{prefix}batch {batch_idx:>4}/{total_batches} | loss~{np.mean(losses[-min(len(losses), 20):]):.4f}")
    return float(np.mean(losses)) if losses else float("nan")


def _build_bigram_baseline(sequences: list[list[int]], pad_id: int) -> dict[int, Counter[int]]:
    counts: dict[int, Counter[int]] = defaultdict(Counter)
    for seq in sequences:
        clean = [token for token in seq if token != pad_id]
        for prev_token, next_token in zip(clean[:-1], clean[1:]):
            counts[int(prev_token)][int(next_token)] += 1
    return counts


def _bigram_perplexity(sequences: list[list[int]], model_counts: dict[int, Counter[int]], vocab_size: int, pad_id: int) -> float:
    total_nll = 0.0
    total_tokens = 0
    alpha = 1.0
    for seq in sequences:
        clean = [token for token in seq if token != pad_id]
        for prev_token, next_token in zip(clean[:-1], clean[1:]):
            next_counts = model_counts.get(int(prev_token), Counter())
            denom = sum(next_counts.values()) + alpha * vocab_size
            prob = (next_counts.get(int(next_token), 0) + alpha) / max(denom, 1e-9)
            total_nll += -math.log(max(prob, 1e-12))
            total_tokens += 1
    if total_tokens == 0:
        return float("inf")
    return float(math.exp(total_nll / total_tokens))


def train_task3(
    *,
    maestro_root: Path = MAESTRO_ROOT,
    maestro_csv: Path = MAESTRO_CSV,
    checkpoint_path: Path = TASK3.checkpoint_path,
    loss_curve_path: Path = TASK3.loss_curve_path,
    perplexity_curve_path: Path = TASK3.perplexity_curve_path,
    report_json: Path = TASK3.report_json,
    generated_dir: Path = TASK3.generated_dir,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 3e-4,
    d_model: int = 256,
    nhead: int = 8,
    num_layers: int = 4,
    dim_feedforward: int = 1024,
    dropout: float = 0.1,
    max_seq_len: int = 2048,
    chunk_length: int = 512,
    stride: int = 256,
    limit_files: Optional[int] = None,
    rebuild_data: bool = False,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    patience: int = 5,
    use_scheduler: bool = True,
    log_every: int = 0,
    generation_samples: int = 10,
    generation_tokens: int = 1024,
    temperature: float = 1.0,
    top_k: int = 32,
) -> dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    data_stats = _ensure_task3_data(
        maestro_root=maestro_root,
        maestro_csv=maestro_csv,
        limit_files=limit_files,
        rebuild=rebuild_data,
        chunk_length=chunk_length,
        stride=stride,
    )
    metadata = load_task3_metadata(TASK3.metadata_json)
    tokenizer = load_task3_tokenizer(TASK3.tokenizer_config_json)
    pad_id = int(metadata["pad_id"])
    bos_id = int(metadata["bos_id"])
    eos_id = int(metadata["eos_id"])
    vocab_size = int(metadata["vocab_size"])
    composer_to_id = {str(k): int(v) for k, v in metadata["composer_to_id"].items()}
    genre_vocab_size = max(composer_to_id.values()) + 1 if composer_to_id else 1

    train_ds = Task3TokenDataset(TASK3.train_jsonl)
    val_ds = Task3TokenDataset(TASK3.val_jsonl)
    train_loader = _prepare_loader(train_ds, batch_size=batch_size, shuffle=True, pad_id=pad_id)
    val_loader = _prepare_loader(val_ds, batch_size=batch_size, shuffle=False, pad_id=pad_id)

    hparams = TransformerHParams(
        vocab_size=vocab_size,
        genre_vocab_size=genre_vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_seq_len=max_seq_len,
        pad_token_id=pad_id,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
    )
    model = DecoderOnlyTransformer(hparams).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")
    best_epoch = 0
    patience_left = patience
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    loss_curve_path.parent.mkdir(parents=True, exist_ok=True)
    perplexity_curve_path.parent.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=dev,
            pad_id=pad_id,
            log_every=log_every,
            label=f"epoch {epoch:03d} train",
        )
        val_loss = _run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=dev,
            pad_id=pad_id,
            log_every=0,
            label=f"epoch {epoch:03d} val",
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if scheduler is not None:
            scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            patience_left = patience
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "hparams": asdict(hparams),
                    "tokenizer_config": metadata["token_config"],
                    "composer_to_id": composer_to_id,
                    "pad_id": pad_id,
                    "bos_id": bos_id,
                    "eos_id": eos_id,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "epoch": epoch,
                    "data_stats": data_stats,
                },
                checkpoint_path,
            )
        else:
            patience_left -= 1

        print(
            f"Epoch {epoch:03d}/{epochs} | train={train_loss:.4f} val={val_loss:.4f} | best_val={best_val:.4f}"
        )
        if patience > 0 and patience_left <= 0:
            print(f"Early stopping: no validation improvement for {patience} epochs")
            break

    epochs_axis = list(range(1, len(train_losses) + 1))
    train_perplexities = [float(math.exp(loss)) for loss in train_losses]
    val_perplexities = [float(math.exp(loss)) for loss in val_losses]

    plt.figure(figsize=(8, 4))
    plt.plot(epochs_axis, train_perplexities, label="train", marker="o", linewidth=2)
    plt.plot(epochs_axis, val_perplexities, label="validation", marker="o", linewidth=2, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title("Task 3: Transformer Perplexity")
    plt.xticks(epochs_axis)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(perplexity_curve_path, dpi=200)
    if loss_curve_path != perplexity_curve_path:
        plt.savefig(loss_curve_path, dpi=200)
    plt.close()

    train_sequences = [record["token_ids"] for record in train_ds.records]
    val_sequences = [record["token_ids"] for record in val_ds.records]
    markov_counts = _build_bigram_baseline(train_sequences, pad_id)
    markov_val_perplexity = _bigram_perplexity(val_sequences, markov_counts, vocab_size, pad_id)
    transformer_val_perplexity = float(math.exp(best_val)) if np.isfinite(best_val) else float("inf")

    model.eval()
    generated_files: list[dict[str, object]] = []
    reference_pitch_hist = aggregate_pitch_class_histogram(
        (
            maestro_root / str(row["midi_filename"])
            for _, row in load_maestro_split(maestro_csv).query("split == 'validation'").iterrows()
        ),
        normalize=True,
    )
    composer_names = sorted(composer_to_id, key=lambda name: composer_to_id[name])
    for idx in range(generation_samples):
        composer_name = composer_names[idx % len(composer_names)] if composer_names else "unknown"
        genre_id = composer_to_id.get(composer_name, 0)
        start_tokens = torch.tensor([bos_id], dtype=torch.long, device=dev).unsqueeze(0)
        genre_tensor = torch.tensor([genre_id], dtype=torch.long, device=dev)
        generated = model.generate(
            start_tokens,
            genre_ids=genre_tensor,
            max_new_tokens=generation_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        token_ids = generated.squeeze(0).tolist()
        if eos_id not in token_ids:
            token_ids.append(eos_id)
        midi_path = generated_dir / f"task3_sample_{idx+1:02d}.midi"
        score = tokenizer.decode([token_ids])
        score.dump_midi(str(midi_path.resolve()))
        validation = validate_midi(midi_path)
        metrics = evaluate_midi_file(midi_path, reference_pitch_histogram=reference_pitch_hist)
        generated_files.append(
            {
                "file": str(midi_path),
                "composer": composer_name,
                "note_count": validation.note_count,
                "duration_seconds": validation.duration_seconds,
                "valid": validation.ok,
                **metrics,
            }
        )

    generated_summary = summarize_generated_directory(
        generated_dir,
        reference_pitch_histogram=reference_pitch_hist,
        glob_pattern="task3_sample_*.midi",
    )
    report = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "transformer_val_perplexity": transformer_val_perplexity,
        "markov_baseline_val_perplexity": markov_val_perplexity,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_perplexities": train_perplexities,
        "val_perplexities": val_perplexities,
        "data_stats": data_stats,
        "hparams": asdict(hparams),
        "generated_files": generated_files,
        "generated_summary": generated_summary,
    }
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Task 3 Transformer model")
    p.add_argument("--maestro_root", type=Path, default=MAESTRO_ROOT)
    p.add_argument("--maestro_csv", type=Path, default=MAESTRO_CSV)
    p.add_argument("--checkpoint", type=Path, default=TASK3.checkpoint_path)
    p.add_argument("--loss_curve", type=Path, default=TASK3.loss_curve_path)
    p.add_argument("--report_json", type=Path, default=TASK3.report_json)
    p.add_argument("--generated_dir", type=Path, default=TASK3.generated_dir)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--dim_feedforward", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--chunk_length", type=int, default=512)
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--limit_files", type=int, default=None)
    p.add_argument("--rebuild_data", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--no_scheduler", action="store_true")
    p.add_argument("--log_every", type=int, default=0)
    p.add_argument("--generation_samples", type=int, default=10)
    p.add_argument("--generation_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=32)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    train_task3(
        maestro_root=args.maestro_root,
        maestro_csv=args.maestro_csv,
        checkpoint_path=args.checkpoint,
        loss_curve_path=args.loss_curve,
        report_json=args.report_json,
        generated_dir=args.generated_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        chunk_length=args.chunk_length,
        stride=args.stride,
        limit_files=args.limit_files,
        rebuild_data=args.rebuild_data,
        seed=args.seed,
        device=args.device,
        patience=args.patience,
        use_scheduler=not bool(args.no_scheduler),
        log_every=args.log_every,
        generation_samples=args.generation_samples,
        generation_tokens=args.generation_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
