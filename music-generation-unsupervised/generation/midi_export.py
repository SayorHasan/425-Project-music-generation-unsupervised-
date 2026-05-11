

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pretty_midi


@dataclass(frozen=True)
class MidiValidationResult:
	ok: bool
	duration_seconds: float
	note_count: int


def piano_roll_to_midi(
	piano_roll_t88: np.ndarray,
	*,
	out_path: Union[str, Path],
	fs: int,
	pitch_min: int = 21,
	velocity: int = 80,
	velocity_variation: int = 0,
) -> Path:
	"""Write a binary piano-roll (T, 88) to a MIDI file.
	
	Args:
		piano_roll_t88: Binary piano roll (T, 88)
		out_path: Output MIDI file path
		fs: Sample rate (frames per second)
		pitch_min: Minimum MIDI pitch (default 21 for A0)
		velocity: Base note velocity (default 80)
		velocity_variation: Amount of velocity variation per pitch (0-40)
	"""

	pr = np.asarray(piano_roll_t88)
	if pr.ndim != 2 or pr.shape[1] != 88:
		raise ValueError(f"Expected piano_roll shape (T, 88), got {pr.shape}")
	pr = (pr > 0).astype(np.uint8)

	active_frames = np.where(pr.sum(axis=1) > 0)[0]
	if active_frames.size > 0:
		first_active = int(active_frames[0])
		if first_active > 0:
			pr = pr[first_active:]

	instrument = pretty_midi.Instrument(program=0, name="piano")

	frame_dur = 1.0 / float(fs)
	T = pr.shape[0]
	
	# Generate per-pitch velocity variations
	if velocity_variation > 0:
		pitch_velocities = np.random.randint(-velocity_variation, velocity_variation + 1, size=88)
		pitch_velocities = np.clip(pitch_velocities + velocity, 20, 127)
	else:
		pitch_velocities = np.full(88, velocity)

	# Create notes by run-length encoding active frames per pitch.
	for p in range(88):
		active = pr[:, p]
		if active.max() == 0:
			continue

		# Find transitions
		onsets = np.where((active[1:] == 1) & (active[:-1] == 0))[0] + 1
		offsets = np.where((active[1:] == 0) & (active[:-1] == 1))[0] + 1
		if active[0] == 1:
			onsets = np.insert(onsets, 0, 0)
		if active[-1] == 1:
			offsets = np.append(offsets, T)

		for start_idx, end_idx in zip(onsets, offsets):
			start = start_idx * frame_dur
			end = max((end_idx * frame_dur), start + frame_dur)
			note = pretty_midi.Note(
				velocity=int(pitch_velocities[p]),
				pitch=int(pitch_min + p),
				start=float(start),
				end=float(end),
			)
			instrument.notes.append(note)

	midi = pretty_midi.PrettyMIDI()
	midi.instruments.append(instrument)

	out_path = Path(out_path)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	midi.write(str(out_path))
	return out_path


def validate_midi(
	midi_path: Union[str, Path],
	*,
	min_duration_seconds: float = 5.0,
	min_notes: int = 50,
) -> MidiValidationResult:
	"""Check loadability, duration, and note count."""

	try:
		pm = pretty_midi.PrettyMIDI(str(midi_path))
	except Exception:
		return MidiValidationResult(ok=False, duration_seconds=0.0, note_count=0)

	duration = float(pm.get_end_time())
	note_count = int(sum(len(inst.notes) for inst in pm.instruments))
	ok = (duration >= min_duration_seconds) and (note_count >= min_notes)
	return MidiValidationResult(ok=ok, duration_seconds=duration, note_count=note_count)
