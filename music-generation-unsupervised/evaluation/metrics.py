"""Metric aggregation utilities (pitch hist similarity, rhythm diversity, repetition ratio, etc.)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi

from evaluation.pitch_histogram import pitch_histogram_similarity_from_midi
from evaluation.rhythm_score import rhythm_diversity_score


def repetition_ratio_from_midi(midi_path: Path | str, *, n: int = 4) -> float:
	try:
		pm = pretty_midi.PrettyMIDI(str(midi_path))
	except Exception:
		return 1.0
	notes = [note for inst in pm.instruments for note in inst.notes]
	notes.sort(key=lambda note: note.start)
	pitches = [int(note.pitch) for note in notes]
	if len(pitches) < n + 1:
		return 1.0
	ngrams = [tuple(pitches[i : i + n]) for i in range(len(pitches) - n + 1)]
	if not ngrams:
		return 1.0
	counts = {}
	for gram in ngrams:
		counts[gram] = counts.get(gram, 0) + 1
	repeated = sum(1 for count in counts.values() if count > 1)
	return float(repeated / len(ngrams))


def note_density_from_midi(midi_path: Path | str) -> float:
	try:
		pm = pretty_midi.PrettyMIDI(str(midi_path))
	except Exception:
		return 0.0
	note_count = sum(len(inst.notes) for inst in pm.instruments)
	duration = float(pm.get_end_time())
	return float(note_count / duration) if duration > 0 else 0.0


def evaluate_midi_file(
	midi_path: Path | str,
	*,
	reference_pitch_histogram: np.ndarray | None = None,
) -> dict[str, float]:
	metrics = {
		"repetition_ratio": repetition_ratio_from_midi(midi_path),
		"rhythm_diversity": 0.0,
		"note_density": note_density_from_midi(midi_path),
	}
	try:
		metrics["rhythm_diversity"] = rhythm_diversity_score(midi_path)
	except Exception:
		metrics["rhythm_diversity"] = 0.0
	if reference_pitch_histogram is not None:
		try:
			metrics["pitch_histogram_similarity"] = pitch_histogram_similarity_from_midi(
				midi_path,
				reference_pitch_histogram,
			)
		except Exception:
			metrics["pitch_histogram_similarity"] = 0.0
	return metrics


def summarize_generated_directory(
	output_dir: Path | str,
	*,
	reference_pitch_histogram: np.ndarray | None = None,
	glob_pattern: str = "*.midi",
) -> dict[str, object]:
	output_dir = Path(output_dir)
	files = sorted(output_dir.glob(glob_pattern))
	records = []
	for midi_path in files:
		try:
			pm = pretty_midi.PrettyMIDI(str(midi_path))
			notes = [note for inst in pm.instruments for note in inst.notes]
			note_count = len(notes)
			duration = float(pm.get_end_time())
		except Exception:
			note_count = 0
			duration = 0.0
		metrics = evaluate_midi_file(midi_path, reference_pitch_histogram=reference_pitch_histogram)
		records.append(
			{
				"file": str(midi_path),
				"note_count": note_count,
				"duration_seconds": duration,
				**metrics,
			}
		)
	means = {key: float(np.mean([record[key] for record in records])) for key in ("repetition_ratio", "rhythm_diversity", "note_density") if records}
	if reference_pitch_histogram is not None and records:
		means["pitch_histogram_similarity"] = float(np.mean([record.get("pitch_histogram_similarity", 0.0) for record in records]))
	return {"files": records, "means": means, "num_files": len(records)}
