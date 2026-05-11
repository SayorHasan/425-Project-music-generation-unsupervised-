# Unsupervised Neural Network for Multi-Genre Music Generation

## Project Overview

This repository implements the full course project pipeline for MAESTRO-based symbolic music generation. The work is organized into three completed modeling tasks:

1. Task 1: LSTM autoencoder on binary piano-roll windows.
2. Task 2: LSTM variational autoencoder with KL annealing.
3. Task 3: decoder-only Transformer on REMI token sequences.

The code also includes evaluation utilities, MIDI export helpers, preprocessing scripts, and generated outputs for each task.

## Dataset and Preprocessing

The project uses the MAESTRO v3.0.0 MIDI split exactly as provided. The split is not re-randomized, which avoids leakage between train, validation, and test sets.

Two preprocessing pipelines are implemented:

- Binary piano-roll windows for Tasks 1 and 2.
- REMI token sequences for Task 3.

The piano-roll pipeline loads MIDI with pretty_midi, keeps the 88-key piano range, binarizes active cells, segments windows, and filters near-silent samples. The token pipeline uses miditok REMI, builds JSONL datasets, and keeps composer metadata for genre conditioning.

## Task 1: LSTM Autoencoder

Task 1 learns a compact latent representation of sparse piano-roll windows.

What is implemented:

- Two-layer LSTM encoder and decoder.
- Latent bottleneck for reconstruction.
- Class-imbalance-aware reconstruction loss.
- MIDI generation from random latent vectors.

Deliverables present in the repo:

- Checkpoint: outputs/task1_autoencoder.pt
- Loss curve: outputs/plots/task1_loss_curve.png
- Generated MIDI samples: outputs/generated_midis/task1/

## Task 2: LSTM VAE

Task 2 extends the autoencoder by learning a Gaussian latent space and using KL annealing to avoid posterior collapse.

What is implemented:

- Mean and log-variance heads on top of the encoder.
- Reparameterization trick.
- KL warmup schedule.
- Diverse MIDI generation from sampled latent vectors.

Deliverables present in the repo:

- Checkpoint: outputs/task2_vae.pt
- Loss curve: outputs/plots/task2_loss_curve.png
- Generated MIDI samples: outputs/generated_midis/task2/

## Task 3: Transformer Generator

Task 3 models music as an autoregressive token sequence problem.

What is implemented:

- REMI tokenizer and token dataset builder.
- Decoder-only Transformer with causal masking.
- Genre/composer conditioning through embedding lookup.
- Validation perplexity measurement.
- Baseline comparison using a Markov-style next-token model.
- Long-sequence MIDI generation.

Deliverables present in the repo:

- Checkpoint: outputs/task3_transformer.pt
- Perplexity curve: outputs/plots/task3_perplexity_curve.png
- Legacy plot path: outputs/plots/task3_loss_curve.png
- Evaluation report: evaluation/task3_report.json
- Generated MIDI samples: outputs/generated_midis/task3/

Recorded Task 3 evaluation values in the report:

- Transformer validation perplexity: 40.19
- Markov baseline validation perplexity: 46.93

## Evaluation Utilities

The evaluation folder contains the MIDI-level metrics used for the project report:

- Pitch histogram similarity.
- Rhythm diversity.
- Repetition ratio.
- Note density.

The metrics are aggregated over generated sample folders and written into the Task 3 evaluation report.

## Repository File Map

### Core configuration

- src/config.py: project paths, dataset roots, and task-specific artifact locations.

### Preprocessing

- src/preprocessing/piano_roll.py: MAESTRO to binary piano-roll windows for Tasks 1 and 2.
- src/preprocessing/tokenizer.py: MAESTRO to REMI token JSONL for Task 3.
- src/preprocessing/midi_parser.py: MIDI parsing helpers used by the preprocessing code.

### Models

- src/models/autoencoder.py: Task 1 autoencoder model.
- src/models/vae.py: Task 2 VAE model.
- src/models/transformer.py: Task 3 decoder-only Transformer model.
- src/models/diffusion.py: additional model experiments not needed for the main submission.

### Training

- src/training/train_ae.py: Task 1 training entry point.
- src/training/train_vae.py: Task 2 training entry point.
- src/training/train_transformer.py: Task 3 training, evaluation, generation, and report creation.

### Generation and export

- generation/generate_music.py: MIDI generation helpers for piano-roll models.
- generation/sample_latent.py: VAE latent sampling and export.
- generation/midi_export.py: MIDI writing and validation helpers.

### Evaluation

- evaluation/metrics.py: aggregate file-level metrics and directory summaries.
- evaluation/pitch_histogram.py: pitch-class similarity calculation.
- evaluation/rhythm_score.py: rhythm diversity scoring.
- evaluation/task3_report.json: saved Task 3 results.

### Outputs

- outputs/task1_autoencoder.pt: Task 1 checkpoint.
- outputs/task2_vae.pt: Task 2 checkpoint.
- outputs/task3_transformer.pt: Task 3 checkpoint.
- outputs/plots/: loss and perplexity plots.
- outputs/generated_midis/: generated samples for all three tasks.

## What To Show The Teacher

If you want to demonstrate that the work was done independently, present the project in this order:

1. Show the MAESTRO split and explain that the provided train/validation/test split was preserved.
2. Show preprocessing code for both representations: piano-roll for Tasks 1 and 2, REMI tokens for Task 3.
3. Show the three model files and explain the architectural differences:
   - autoencoder: compressed reconstruction,
   - VAE: probabilistic latent space with KL annealing,
   - Transformer: autoregressive token prediction with causal masking.
4. Open the training scripts and explain how each one saves checkpoints, plots, generated MIDI, and reports.
5. Open the generated output folders and play or inspect a few samples.
6. Open evaluation/task3_report.json and point out the validation perplexity and baseline comparison.

## Short Presentation Script

You can summarize the project to your teacher like this:

"I built the pipeline from MAESTRO data loading to preprocessing, model training, generation, and evaluation. Tasks 1 and 2 use sparse piano-roll windows with a class-imbalance-aware loss, while Task 3 uses REMI tokenization and a causal Transformer for autoregressive generation. I saved checkpoints, plots, generated MIDI samples, and an evaluation report comparing the Transformer against a baseline."

## Final Deliverables Checklist

- Task 1 model, plot, and 5 MIDI samples: done.
- Task 2 model, plot, and 8 MIDI samples: done.
- Task 3 model, perplexity evaluation, baseline comparison, and 10 MIDI samples: done.
- Report and repo documentation: this file.

## Notes

The Task 3 plot is saved as a perplexity curve so it remains visible even for short smoke runs. The report and the code are aligned with the implementation guide: no random re-splitting of MAESTRO, no missing causal mask, and evaluation is performed directly on generated MIDI files.