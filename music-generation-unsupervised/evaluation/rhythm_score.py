"""Rhythm diversity metric."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi


def onset_times_from_midi(midi_path: Path | str) -> np.ndarray:
	pm = pretty_midi.PrettyMIDI(str(midi_path))
	onsets = [note.start for inst in pm.instruments for note in inst.notes]
	if not onsets:
		return np.zeros(0, dtype=np.float64)
	return np.asarray(sorted(onsets), dtype=np.float64)


def rhythm_diversity_score(midi_path: Path | str, *, bin_size_seconds: float = 0.05) -> float:
	onsets = onset_times_from_midi(midi_path)
	if onsets.size < 2:
		return 0.0
	bins = np.floor(onsets / max(bin_size_seconds, 1e-6)).astype(int)
	counts = np.bincount(bins)
	counts = counts[counts > 0]
	if counts.size == 0:
		return 0.0
	prob = counts / counts.sum()
	entropy = -np.sum(prob * np.log(prob + 1e-12))
	max_entropy = np.log(len(prob) + 1e-12)
	if max_entropy <= 0.0:
		return 0.0
	return float(np.clip(entropy / max_entropy, 0.0, 1.0))
