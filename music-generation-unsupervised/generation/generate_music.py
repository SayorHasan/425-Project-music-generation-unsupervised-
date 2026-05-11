

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import pretty_midi

from generation.midi_export import piano_roll_to_midi, validate_midi
from src.config import PIANO_ROLL, TASK1
from src.models.autoencoder import AutoencoderHParams, LSTMAutoencoder


def load_task1_checkpoint(path: Path, device: torch.device) -> tuple[LSTMAutoencoder, dict]:
	ckpt = torch.load(path, map_location=device)
	hparams = AutoencoderHParams(**ckpt["hparams"])
	model = LSTMAutoencoder(hparams).to(device)
	model.load_state_dict(ckpt["state_dict"])
	model.eval()
	return model, ckpt


def generate_task1(
	*,
	checkpoint: Path,
	out_dir: Path,
	num_samples: int,
	threshold: float,
	auto_threshold: bool,
	seed: int,
	device: str,
	seq_len: int,
) -> None:
	dev = torch.device(device)
	torch.manual_seed(seed)
	np.random.seed(seed)

	model, ckpt = load_task1_checkpoint(checkpoint, dev)
	latent_dim = model.hparams.latent_dim

	out_dir.mkdir(parents=True, exist_ok=True)

	def repetition_ratio(midi_path: Path, n: int = 4) -> float:
		try:
			pm = pretty_midi.PrettyMIDI(str(midi_path))
		except Exception:
			return 1.0
		notes = []
		for inst in pm.instruments:
			notes.extend(inst.notes)
		notes.sort(key=lambda x: x.start)
		pitches = [int(n_.pitch) for n_ in notes]
		if len(pitches) < n + 1:
			return 1.0
		ngrams = [tuple(pitches[i : i + n]) for i in range(0, len(pitches) - n + 1)]
		total = len(ngrams)
		if total == 0:
			return 1.0
		counts = {}
		for g in ngrams:
			counts[g] = counts.get(g, 0) + 1
		repeated = sum(1 for c in counts.values() if c > 1)
		return float(repeated / total)

	def try_generate_with_threshold(th: float, target_note_range=(80, 1500)) -> tuple[int, float]:
		"""Return (successes, avg_rep_ratio) over a few attempts."""
		reps = []
		successes = 0
		for _ in range(3):
			z = torch.randn(1, latent_dim, device=dev)
			with torch.no_grad():
				logits = model.decode(z, seq_len=seq_len)
				probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
			piano_roll = (probs >= th).astype(np.uint8)
			tmp_path = out_dir / "_tmp_threshold_test.midi"
			piano_roll_to_midi(
				piano_roll,
				out_path=tmp_path,
				fs=int(ckpt.get("fs", PIANO_ROLL.fs)),
				pitch_min=int(ckpt.get("pitch_min", PIANO_ROLL.pitch_min)),
				velocity=int(ckpt.get("velocity", PIANO_ROLL.default_velocity)),
			)
			v = validate_midi(tmp_path)
			if v.ok and (target_note_range[0] <= v.note_count <= target_note_range[1]):
				rr = repetition_ratio(tmp_path)
				reps.append(rr)
				successes += 1
			tmp_path.unlink(missing_ok=True)
		if successes == 0:
			return 0, 1.0
		return successes, float(np.mean(reps))

	if auto_threshold:
		candidates = [
			0.05,
			0.06,
			0.07,
			0.08,
			0.09,
			0.10,
			0.12,
			0.14,
			0.16,
			0.18,
			0.21,
			0.24,
			0.27,
			0.30,
			0.33,
		]
		best = None
		for th in candidates:
			succ, rep = try_generate_with_threshold(th)
			# score: prefer more successes, lower repetition
			score = (succ * 10.0) - (rep * 10.0)
			if best is None or score > best[0]:
				best = (score, th, succ, rep)
		if best is not None:
			_, threshold, succ, rep = best
			print(f"Auto-threshold selected {threshold:.2f} (succ={succ}/3, rep={rep:.3f})")

	made = 0
	attempts = 0
	while made < num_samples:
		attempts += 1
		z = torch.randn(1, latent_dim, device=dev)
		with torch.no_grad():
			logits = model.decode(z, seq_len=seq_len)
			probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (T, 88)
		piano_roll = (probs >= threshold).astype(np.uint8)

		out_path = out_dir / f"task1_sample_{made+1:02d}.midi"
		piano_roll_to_midi(
			piano_roll,
			out_path=out_path,
			fs=int(ckpt.get("fs", PIANO_ROLL.fs)),
			pitch_min=int(ckpt.get("pitch_min", PIANO_ROLL.pitch_min)),
			velocity=int(ckpt.get("velocity", PIANO_ROLL.default_velocity)),
		)
		v = validate_midi(out_path)
		if not v.ok:
			out_path.unlink(missing_ok=True)
			# Try a new sample (or adjust threshold if repeatedly failing)
			if attempts > 50 and threshold > 0.1:
				threshold *= 0.9
			continue

		rr = repetition_ratio(out_path)
		if rr > 0.80:
			out_path.unlink(missing_ok=True)
			continue
		made += 1


def _parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Generate MIDI samples (Task 1)")
	p.add_argument("--checkpoint", type=Path, default=TASK1.checkpoint_path)
	p.add_argument("--out_dir", type=Path, default=TASK1.generated_dir)
	p.add_argument("--num_samples", type=int, default=5)
	p.add_argument("--threshold", type=float, default=0.30)
	p.add_argument(
		"--auto_threshold",
		action="store_true",
		help="Try multiple thresholds and pick one that avoids sparse/repetitive outputs",
	)
	p.add_argument("--seed", type=int, default=42)
	p.add_argument(
		"--device",
		type=str,
		default="cuda" if torch.cuda.is_available() else "cpu",
	)
	p.add_argument("--seq_len", type=int, default=PIANO_ROLL.window_steps)
	return p.parse_args()


def main() -> None:
	args = _parse_args()
	generate_task1(
		checkpoint=args.checkpoint,
		out_dir=args.out_dir,
		num_samples=args.num_samples,
		threshold=args.threshold,
		auto_threshold=bool(args.auto_threshold),
		seed=args.seed,
		device=args.device,
		seq_len=args.seq_len,
	)


if __name__ == "__main__":
	main()
