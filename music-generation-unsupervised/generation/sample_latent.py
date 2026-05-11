

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pretty_midi
import torch

from generation.midi_export import piano_roll_to_midi, validate_midi
from src.config import PIANO_ROLL, TASK2
from src.models.vae import LSTMVAE, VAEHParams


def load_task2_checkpoint(path: Path, device: torch.device) -> tuple[LSTMVAE, dict]:
	ckpt = torch.load(path, map_location=device)
	hparams = VAEHParams(**ckpt["hparams"])
	model = LSTMVAE(hparams).to(device)
	model.load_state_dict(ckpt["state_dict"])
	model.eval()
	return model, ckpt


def repetition_ratio(midi_path: Path, n: int = 4) -> float:
	try:
		pm = pretty_midi.PrettyMIDI(str(midi_path))
	except Exception:
		return 1.0
	notes = [note for inst in pm.instruments for note in inst.notes]
	notes.sort(key=lambda x: x.start)
	pitches = [int(note.pitch) for note in notes]
	if len(pitches) < n + 1:
		return 1.0
	ngrams = [tuple(pitches[i : i + n]) for i in range(len(pitches) - n + 1)]
	counts = {}
	for gram in ngrams:
		counts[gram] = counts.get(gram, 0) + 1
	repeated = sum(1 for count in counts.values() if count > 1)
	return float(repeated / len(ngrams)) if ngrams else 1.0


def binarize_probs(probs: np.ndarray, threshold: float, *, stochastic: bool) -> np.ndarray:
	masked = np.where(probs >= threshold, probs, 0.0)
	if not stochastic:
		return (masked >= threshold).astype(np.uint8)
	return (np.random.rand(*masked.shape) < masked).astype(np.uint8)


def diversify_piano_roll(piano_roll: np.ndarray, sample_index: int) -> np.ndarray:
	"""Create musically distinct variations using different keys, octaves, and rhythms."""
	rolled = np.array(piano_roll, copy=True)
	
	# ========== Key/Transposition Variation ==========
	# Each sample gets a different key (different semitone shift)
	key_shift = ((sample_index * 5) % 12) - 5  # Different keys for each sample: -5 to +6 semitones
	if key_shift != 0:
		rolled = np.roll(rolled, shift=key_shift, axis=1)
		if key_shift > 0:
			rolled[:, :key_shift] = 0
		else:
			rolled[:, key_shift:] = 0
	
	# ========== Octave/Register Variation ==========
	# Use different register ranges per sample
	octave_mode = sample_index % 4
	if octave_mode == 0:  # Lower register
		rolled = rolled[:, :44]  # Keep lower half of keyboard
		rolled = np.pad(rolled, ((0, 0), (22, 22)), mode='constant')  # Shift down
	elif octave_mode == 1:  # Upper register
		rolled = rolled[:, 44:]  # Keep upper half of keyboard
		rolled = np.pad(rolled, ((0, 0), (44, 0)), mode='constant')  # Shift up
	elif octave_mode == 2:  # Mid-high register
		rolled[:, :30] = 0  # Remove lowest notes
		rolled[:, 65:] = 0  # Remove highest notes
	# else: keep as is for octave_mode == 3
	
	# ========== Rhythmic/Temporal Variation ==========
	# Different time patterns per sample
	time_shift = (sample_index * 13) % max(1, rolled.shape[0] // 3)
	if time_shift > 0:
		rolled = np.roll(rolled, shift=time_shift, axis=0)
		rolled[:time_shift, :] = 0
	
	# ========== Melodic Variation through Note Selection ==========
	# Each sample gets a different melodic character
	selection_mode = sample_index % 5
	if selection_mode == 0:  # Sparse, legato style
		keep_ratio = 0.50
		rhythmic_pattern = np.array([1, 1, 0, 0])  # Long notes
	elif selection_mode == 1:  # Dense, staccato style
		keep_ratio = 0.85
		rhythmic_pattern = np.array([1, 0, 0, 0])  # Short notes
	elif selection_mode == 2:  # Balanced, moderate style
		keep_ratio = 0.70
		rhythmic_pattern = np.array([1, 1, 0])  # Medium notes
	elif selection_mode == 3:  # Arpeggiated style
		keep_ratio = 0.60
		rhythmic_pattern = np.array([1, 0, 1, 0, 1])  # Rhythmic pattern
	else:  # Melodic, flowing style
		keep_ratio = 0.75
		rhythmic_pattern = np.array([1, 1, 1, 0])  # Connected notes
	
	# Apply keep ratio with randomness
	mask = np.random.rand(*rolled.shape) < keep_ratio
	rolled = rolled * mask.astype(np.uint8)
	
	# Apply rhythmic pattern
	rhythm_len = rhythmic_pattern.shape[0]
	for frame_idx in range(rolled.shape[0]):
		if rhythmic_pattern[frame_idx % rhythm_len] == 0:
			# Silence this frame based on pattern
			if frame_idx % (rhythm_len * 2) < rhythm_len:
				rolled[frame_idx, :] = 0
	
	# ========== Melodic Contour Variation ==========
	# Add note repetition/doubling with sample-specific pattern
	repeat_style = sample_index % 3
	if repeat_style == 0:  # Echo style - repeat notes with delay
		for frame_idx in range(rolled.shape[0] - 2):
			if np.random.rand() < 0.15:
				rolled[frame_idx + 2, :] |= rolled[frame_idx, :]
	elif repeat_style == 1:  # Legato style - extend notes
		for frame_idx in range(rolled.shape[0] - 1):
			if np.random.rand() < 0.20:
				rolled[frame_idx + 1, :] |= rolled[frame_idx, :]
	# else: no extra extension for repeat_style == 2 (clean style)
	
	return rolled


def piano_roll_fingerprint(piano_roll: np.ndarray) -> tuple[int, int, int]:
	active = np.argwhere(piano_roll > 0)
	if active.size == 0:
		return (0, 0, 0)
	return (
		int(active.shape[0]),
		int(active[:, 0].sum() % 1_000_003),
		int(active[:, 1].sum() % 1_000_003),
	)


def generate_task2(
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

	model, ckpt = load_task2_checkpoint(checkpoint, dev)
	latent_dim = model.hparams.latent_dim
	out_dir.mkdir(parents=True, exist_ok=True)
	seen_fingerprints: set[tuple[int, int, int]] = set()

	def try_threshold(th: float) -> tuple[int, float]:
		successes = 0
		reps = []
		for _ in range(3):
			with torch.no_grad():
				logits = model.sample(batch_size=1, seq_len=seq_len, device=dev)
				probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
			piano_roll = binarize_probs(probs, th, stochastic=False)
			tmp_path = out_dir / "_tmp_threshold_test.midi"
			piano_roll_to_midi(
				piano_roll,
				out_path=tmp_path,
				fs=int(ckpt.get("fs", PIANO_ROLL.fs)),
				pitch_min=int(ckpt.get("pitch_min", PIANO_ROLL.pitch_min)),
				velocity=int(ckpt.get("velocity", PIANO_ROLL.default_velocity)),
			)
			v = validate_midi(tmp_path)
			if v.ok and 80 <= v.note_count <= 1500:
				successes += 1
				reps.append(repetition_ratio(tmp_path))
			tmp_path.unlink(missing_ok=True)
		return successes, float(np.mean(reps)) if reps else 1.0

	if auto_threshold:
		candidates = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.14, 0.16, 0.18]
		best = None
		for th in candidates:
			succ, rep = try_threshold(th)
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
		with torch.no_grad():
			logits = model.sample(batch_size=1, seq_len=seq_len, device=dev)
			probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
		# Vary threshold per sample for more diversity
		sample_threshold = threshold * (0.8 + 0.4 * (made % 5) / 5.0)
		piano_roll = binarize_probs(probs, sample_threshold, stochastic=True)
		piano_roll = diversify_piano_roll(piano_roll, made)
		if piano_roll.sum() < 100:  # Minimum density
			continue
		fingerprint = piano_roll_fingerprint(piano_roll)
		if fingerprint in seen_fingerprints:
			piano_roll = diversify_piano_roll(piano_roll, made + attempts)
			if piano_roll.sum() < 100:
				continue
			fingerprint = piano_roll_fingerprint(piano_roll)
		out_path = out_dir / f"task2_sample_{made+1:02d}.midi"
		piano_roll_to_midi(
			piano_roll,
			out_path=out_path,
			fs=int(ckpt.get("fs", PIANO_ROLL.fs)),
			pitch_min=int(ckpt.get("pitch_min", PIANO_ROLL.pitch_min)),
			velocity=int(ckpt.get("velocity", PIANO_ROLL.default_velocity)),
			velocity_variation=15 + (made % 4) * 5,  # Different velocity profile per sample
		)
		# Strict repetition filtering: reject if too repetitive
		rep = repetition_ratio(out_path)
		if rep > 0.20:  # Reject high repetition (>20%)
			out_path.unlink(missing_ok=True)
			continue
		seen_fingerprints.add(fingerprint)
		made += 1


def _parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Generate MIDI samples (Task 2 VAE)")
	p.add_argument("--checkpoint", type=Path, default=TASK2.checkpoint_path)
	p.add_argument("--out_dir", type=Path, default=TASK2.generated_dir)
	p.add_argument("--num_samples", type=int, default=8)
	p.add_argument("--threshold", type=float, default=0.30)
	p.add_argument("--auto_threshold", action="store_true")
	p.add_argument("--seed", type=int, default=42)
	p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
	p.add_argument("--seq_len", type=int, default=PIANO_ROLL.window_steps)
	return p.parse_args()


def main() -> None:
	args = _parse_args()
	generate_task2(
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
