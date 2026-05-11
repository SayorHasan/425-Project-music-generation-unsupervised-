"""Pitch histogram similarity metric (12 pitch classes)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pretty_midi


def pitch_class_histogram_from_midi(midi_path: Path | str, *, normalize: bool = True) -> np.ndarray:
	pm = pretty_midi.PrettyMIDI(str(midi_path))
	hist = np.zeros(12, dtype=np.float64)
	for inst in pm.instruments:
		for note in inst.notes:
			hist[int(note.pitch) % 12] += 1.0
	if normalize and hist.sum() > 0:
		hist /= hist.sum()
	return hist


def aggregate_pitch_class_histogram(midi_paths: Iterable[Path | str], *, normalize: bool = True) -> np.ndarray:
	hist = np.zeros(12, dtype=np.float64)
	for midi_path in midi_paths:
		hist += pitch_class_histogram_from_midi(midi_path, normalize=False)
	if normalize and hist.sum() > 0:
		hist /= hist.sum()
	return hist


def pitch_histogram_similarity(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
	a = np.asarray(hist_a, dtype=np.float64)
	b = np.asarray(hist_b, dtype=np.float64)
	a_norm = float(np.linalg.norm(a))
	b_norm = float(np.linalg.norm(b))
	if a_norm == 0.0 or b_norm == 0.0:
		return 0.0
	return float(np.dot(a, b) / (a_norm * b_norm))


def pitch_histogram_similarity_from_midi(
	midi_path: Path | str,
	reference_histogram: np.ndarray,
	*,
	normalize: bool = True,
) -> float:
	hist = pitch_class_histogram_from_midi(midi_path, normalize=normalize)
	return pitch_histogram_similarity(hist, reference_histogram)
