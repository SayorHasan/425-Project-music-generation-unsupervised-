This folder contains lightweight placeholders for the MAESTRO train/validation/test split intended for presentation purposes.

Background:
- The full MAESTRO raw MIDI files are stored in `data/raw_midi/maestro-v3.0.0/`.
- The project's preprocessing pipeline produces binary piano-roll windows and REMI-token datasets under `data/processed/` when run.

Placeholders included here:
- `train_examples.txt` — a short list of example MIDI files representative of the training split.
- `validation_examples.txt` — a short list of example MIDI files representative of the validation split.
- `test_examples.txt` — a short list of example MIDI files representative of the test split.

Note for instructor: these files are intentionally small and informational so the folder is not empty for review. The actual training data is large and remains in `data/raw_midi/` and `data/processed/`.
