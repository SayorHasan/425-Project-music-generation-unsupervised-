# music-generation-unsupervised

Course project: Unsupervised Neural Network for Multi-Genre Music Generation CSE425.

## Workspace layout
This repo follows the assignment’s required structure.

## Dataset
- Place your chosen dataset under `data/raw_midi/`.
- If you use MAESTRO, keep the provided `maestro-v3.0.0.csv` split (`train/validation/test`). Do not re-split randomly.

## Setup
1. Install PyTorch from the official selector (varies by CUDA/CPU): https://pytorch.org/get-started/locally/
2. Install the remaining dependencies:
   - `pip install -r requirements.txt`

## Next steps

### Task 1 (Mandatory): LSTM Autoencoder

**A) Preprocess (MAESTRO → binary piano-roll windows)**

From the `music-generation-unsupervised/` folder:

1. Build the window datasets (train/validation/test) using MAESTRO’s provided split:
   - `python -m src.preprocessing.piano_roll`

Outputs:
- `data/processed/task1_train_windows.npy`
- `data/processed/task1_val_windows.npy`
- `data/processed/task1_test_windows.npy`
- `data/processed/task1_stats.json`

Optional speed-up while debugging:
- `python -m src.preprocessing.piano_roll --limit_files 50`

**B) Train the autoencoder**

2. Train and save the best checkpoint (by validation loss) + loss curve plot:
   - `python -m src.training.train_ae --epochs 50`

Outputs:
- `outputs/task1_autoencoder.pt`
- `outputs/plots/task1_loss_curve.png`

**C) Generate MIDI samples (deliverable: 5 samples)**

3. Generate 5 valid MIDI samples from random latent vectors:
   - `python -m generation.generate_music --num_samples 5`

Outputs:
- `outputs/generated_midis/task1/task1_sample_01.midi` … `task1_sample_05.midi`

---

### Tasks 2–4
- Task 2: train the VAE with `python -m src.training.train_vae`
- Generate 8 samples with `python -m generation.sample_latent --auto_threshold --num_samples 8`
- Outputs:
   - `outputs/task2_vae.pt`
   - `outputs/plots/task2_loss_curve.png`
   - `outputs/generated_midis/task2/`
- Task 3: implement in `src/models/transformer.py`, train via `src/training/train_transformer.py`
