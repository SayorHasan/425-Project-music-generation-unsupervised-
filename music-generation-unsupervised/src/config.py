"""Project configuration (paths, constants, hyperparameters).

Keep this file small and importable by scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_MIDI_DIR = DATA_DIR / "raw_midi"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
GENERATED_MIDIS_DIR = OUTPUTS_DIR / "generated_midis"
PLOTS_DIR = OUTPUTS_DIR / "plots"
SURVEY_RESULTS_DIR = OUTPUTS_DIR / "survey_results"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"

# Dataset roots (MAESTRO is included in the parent workspace folder)
WORKSPACE_ROOT = PROJECT_ROOT.parent
MAESTRO_ROOT = WORKSPACE_ROOT / "maestro-v3.0.0"
MAESTRO_CSV = MAESTRO_ROOT / "maestro-v3.0.0.csv"


@dataclass(frozen=True)
class PianoRollConfig:
    fs: int = 16
    window_steps: int = 128
    pitch_min: int = 21
    pitch_max: int = 108

    # Filter out near-silent windows (sparsity mitigation)
    min_active_fraction: float = 0.02

    # MIDI export
    default_velocity: int = 80


PIANO_ROLL = PianoRollConfig()


@dataclass(frozen=True)
class Task1Config:
    train_npy: Path = PROCESSED_DIR / "task1_train_windows.npy"
    val_npy: Path = PROCESSED_DIR / "task1_val_windows.npy"
    test_npy: Path = PROCESSED_DIR / "task1_test_windows.npy"
    stats_json: Path = PROCESSED_DIR / "task1_stats.json"

    checkpoint_path: Path = OUTPUTS_DIR / "task1_autoencoder.pt"
    loss_curve_path: Path = PLOTS_DIR / "task1_loss_curve.png"
    generated_dir: Path = GENERATED_MIDIS_DIR / "task1"


@dataclass(frozen=True)
class Task2Config:
    train_npy: Path = PROCESSED_DIR / "task1_train_windows.npy"
    val_npy: Path = PROCESSED_DIR / "task1_val_windows.npy"
    test_npy: Path = PROCESSED_DIR / "task1_test_windows.npy"

    checkpoint_path: Path = OUTPUTS_DIR / "task2_vae.pt"
    loss_curve_path: Path = PLOTS_DIR / "task2_loss_curve.png"
    generated_dir: Path = GENERATED_MIDIS_DIR / "task2"


@dataclass(frozen=True)
class Task3Config:
    token_dir: Path = PROCESSED_DIR / "task3"
    train_jsonl: Path = token_dir / "task3_train_sequences.jsonl"
    val_jsonl: Path = token_dir / "task3_val_sequences.jsonl"
    test_jsonl: Path = token_dir / "task3_test_sequences.jsonl"
    tokenizer_config_json: Path = token_dir / "task3_tokenizer_config.json"
    metadata_json: Path = token_dir / "task3_metadata.json"

    checkpoint_path: Path = OUTPUTS_DIR / "task3_transformer.pt"
    loss_curve_path: Path = PLOTS_DIR / "task3_loss_curve.png"
    perplexity_curve_path: Path = PLOTS_DIR / "task3_perplexity_curve.png"
    generated_dir: Path = GENERATED_MIDIS_DIR / "task3"
    report_json: Path = EVALUATION_DIR / "task3_report.json"


TASK1 = Task1Config()
TASK2 = Task2Config()
TASK3 = Task3Config()
