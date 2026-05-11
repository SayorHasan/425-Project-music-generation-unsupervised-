"""Piano-roll extraction + windowing for Tasks 1 and 2.

Task 1 requires:
- Convert MAESTRO MIDI files to binary piano-roll
- Normalize timing resolution (fs=16 recommended)
- Segment into fixed windows (128 steps recommended)
- Filter sparse windows (to mitigate 97–98% silence)

This module implements that preprocessing for MAESTRO using the provided metadata split.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
import pandas as pd

from src.config import MAESTRO_CSV, MAESTRO_ROOT, PIANO_ROLL, TASK1
from src.preprocessing.midi_parser import load_pretty_midi


def _extract_binary_piano_roll(
	midi_path: Path,
	*,
	fs: int,
	pitch_min: int,
	pitch_max: int,
) -> Optional[np.ndarray]:
	"""Return binary piano-roll of shape (T, 88) as uint8, or None on load failure."""

	pm = load_pretty_midi(midi_path)
	if pm is None:
		return None

	# pretty_midi returns array shaped (128, T)
	roll = pm.get_piano_roll(fs=fs)
	pitch_rows = roll[pitch_min : pitch_max + 1]  # inclusive
	if pitch_rows.size == 0:
		return None
	# (88, T) -> (T, 88)
	roll_t = pitch_rows.T
	# binarize
	roll_bin = (roll_t > 0).astype(np.uint8)
	return roll_bin


def _iter_non_overlapping_windows(roll_t88: np.ndarray, window_steps: int) -> Iterator[np.ndarray]:
	total_steps = int(roll_t88.shape[0])
	usable = total_steps - (total_steps % window_steps)
	for start in range(0, usable, window_steps):
		yield roll_t88[start : start + window_steps]


def _active_fraction(window: np.ndarray) -> float:
	# window is uint8 (0/1)
	return float(window.sum() / window.size)


def load_maestro_split(csv_path: Path = MAESTRO_CSV) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	required_cols = {"split", "midi_filename"}
	missing = required_cols - set(df.columns)
	if missing:
		raise ValueError(f"MAESTRO CSV missing columns: {sorted(missing)}")
	return df


def _iter_split_midi_paths(df: pd.DataFrame, split: str, maestro_root: Path) -> Iterable[Path]:
	split_df = df[df["split"] == split]
	for rel in split_df["midi_filename"].astype(str).tolist():
		yield maestro_root / rel


def count_windows(
	midi_paths: Iterable[Path],
	*,
	fs: int,
	pitch_min: int,
	pitch_max: int,
	window_steps: int,
	min_active_fraction: float,
	limit_files: Optional[int] = None,
) -> tuple[int, dict[str, int]]:
	"""First pass: count how many windows survive filtering."""

	total = 0
	stats = {
		"files_seen": 0,
		"files_loaded": 0,
		"files_failed": 0,
		"windows_total": 0,
		"windows_kept": 0,
		"windows_dropped_sparse": 0,
	}

	for midi_path in midi_paths:
		stats["files_seen"] += 1
		if limit_files is not None and stats["files_seen"] > limit_files:
			break

		if stats["files_seen"] % 50 == 0:
			print(
				f"  count: seen={stats['files_seen']} loaded={stats['files_loaded']} kept={stats['windows_kept']}"
			)

		roll = _extract_binary_piano_roll(
			midi_path, fs=fs, pitch_min=pitch_min, pitch_max=pitch_max
		)
		if roll is None:
			stats["files_failed"] += 1
			continue

		stats["files_loaded"] += 1
		for window in _iter_non_overlapping_windows(roll, window_steps=window_steps):
			stats["windows_total"] += 1
			if _active_fraction(window) < min_active_fraction:
				stats["windows_dropped_sparse"] += 1
				continue
			stats["windows_kept"] += 1
			total += 1
	return total, stats


def write_windows(
	midi_paths: Iterable[Path],
	out_npy: Path,
	*,
	fs: int,
	pitch_min: int,
	pitch_max: int,
	window_steps: int,
	min_active_fraction: float,
	expected_windows: int,
	limit_files: Optional[int] = None,
) -> dict[str, int]:
	"""Second pass: write windows into an .npy memmap file."""

	out_npy.parent.mkdir(parents=True, exist_ok=True)
	mm = np.lib.format.open_memmap(
		out_npy, mode="w+", dtype=np.uint8, shape=(expected_windows, window_steps, 88)
	)

	stats = {
		"files_seen": 0,
		"files_loaded": 0,
		"files_failed": 0,
		"windows_written": 0,
		"windows_dropped_sparse": 0,
	}
	write_idx = 0
	for midi_path in midi_paths:
		stats["files_seen"] += 1
		if limit_files is not None and stats["files_seen"] > limit_files:
			break

		if stats["files_seen"] % 50 == 0:
			print(
				f"  write: seen={stats['files_seen']} loaded={stats['files_loaded']} written={write_idx}"
			)

		roll = _extract_binary_piano_roll(
			midi_path, fs=fs, pitch_min=pitch_min, pitch_max=pitch_max
		)
		if roll is None:
			stats["files_failed"] += 1
			continue
		stats["files_loaded"] += 1

		for window in _iter_non_overlapping_windows(roll, window_steps=window_steps):
			if _active_fraction(window) < min_active_fraction:
				stats["windows_dropped_sparse"] += 1
				continue
			if write_idx >= expected_windows:
				raise RuntimeError(
					f"More windows than expected: write_idx={write_idx} expected={expected_windows}"
				)
			mm[write_idx] = window
			write_idx += 1
	stats["windows_written"] = write_idx

	if write_idx != expected_windows:
		# Shrinking an .npy memmap isn't trivial; keep it strict so results are reproducible.
		raise RuntimeError(
			f"Window count mismatch: wrote {write_idx}, expected {expected_windows}."
		)
	mm.flush()
	return stats


def build_task1_dataset(
	*,
	maestro_root: Path = MAESTRO_ROOT,
	maestro_csv: Path = MAESTRO_CSV,
	out_train: Path = TASK1.train_npy,
	out_val: Path = TASK1.val_npy,
	out_test: Path = TASK1.test_npy,
	out_stats_json: Path = TASK1.stats_json,
	fs: int = PIANO_ROLL.fs,
	window_steps: int = PIANO_ROLL.window_steps,
	pitch_min: int = PIANO_ROLL.pitch_min,
	pitch_max: int = PIANO_ROLL.pitch_max,
	min_active_fraction: float = PIANO_ROLL.min_active_fraction,
	limit_files: Optional[int] = None,
) -> None:
	df = load_maestro_split(maestro_csv)
	out_stats_json.parent.mkdir(parents=True, exist_ok=True)

	splits = {
		"train": (out_train, list(_iter_split_midi_paths(df, "train", maestro_root))),
		"validation": (
			out_val,
			list(_iter_split_midi_paths(df, "validation", maestro_root)),
		),
		"test": (out_test, list(_iter_split_midi_paths(df, "test", maestro_root))),
	}

	all_stats: dict[str, object] = {
		"piano_roll_config": asdict(PIANO_ROLL),
		"maestro_root": str(maestro_root),
		"maestro_csv": str(maestro_csv),
		"splits": {},
	}

	for split_name, (out_path, midi_paths) in splits.items():
		print(f"Split: {split_name} | midi files: {len(midi_paths)}")
		print("Pass 1/2: counting windows...")
		expected, count_stats = count_windows(
			midi_paths,
			fs=fs,
			pitch_min=pitch_min,
			pitch_max=pitch_max,
			window_steps=window_steps,
			min_active_fraction=min_active_fraction,
			limit_files=limit_files,
		)
		print(f"Pass 1 done: expected_windows={expected}")
		print("Pass 2/2: writing windows...")
		write_stats = write_windows(
			midi_paths,
			out_path,
			fs=fs,
			pitch_min=pitch_min,
			pitch_max=pitch_max,
			window_steps=window_steps,
			min_active_fraction=min_active_fraction,
			expected_windows=expected,
			limit_files=limit_files,
		)
		all_stats["splits"][split_name] = {
			"out_file": str(out_path),
			"expected_windows": expected,
			"count_stats": count_stats,
			"write_stats": write_stats,
		}

		print(f"Pass 2 done: wrote_windows={write_stats['windows_written']}")

	out_stats_json.write_text(json.dumps(all_stats, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Build Task 1 piano-roll window dataset (MAESTRO)")
	p.add_argument("--maestro_root", type=Path, default=MAESTRO_ROOT)
	p.add_argument("--maestro_csv", type=Path, default=MAESTRO_CSV)
	p.add_argument("--limit_files", type=int, default=None)
	p.add_argument("--fs", type=int, default=PIANO_ROLL.fs)
	p.add_argument("--window_steps", type=int, default=PIANO_ROLL.window_steps)
	p.add_argument("--min_active_fraction", type=float, default=PIANO_ROLL.min_active_fraction)
	return p.parse_args()


def main() -> None:
	args = _parse_args()
	build_task1_dataset(
		maestro_root=args.maestro_root,
		maestro_csv=args.maestro_csv,
		fs=args.fs,
		window_steps=args.window_steps,
		min_active_fraction=args.min_active_fraction,
		limit_files=args.limit_files,
	)


if __name__ == "__main__":
	main()
