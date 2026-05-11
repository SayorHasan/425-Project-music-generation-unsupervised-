"""Utilities for loading MIDI files safely."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from typing import Union

import pretty_midi


def load_pretty_midi(path: Union[str, Path]) -> Optional[pretty_midi.PrettyMIDI]:
    """Load a MIDI file; return None if parsing fails."""

    try:
        return pretty_midi.PrettyMIDI(str(path))
    except Exception:
        return None
