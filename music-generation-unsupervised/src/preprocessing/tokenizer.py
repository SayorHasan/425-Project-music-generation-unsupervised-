"""Tokenization helpers (miditok) for Transformer-based tasks."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
import pandas as pd
import torch
from miditok import REMI, TokenizerConfig
from torch.utils.data import Dataset

from src.config import MAESTRO_CSV, MAESTRO_ROOT, TASK3


@dataclass(frozen=True)
class Task3TokenConfig:
	pitch_low: int = 21
	pitch_high: int = 109
	num_velocities: int = 32
	chunk_length: int = 512
	stride: int = 256
	min_chunk_length: int = 128
	max_generation_length: int = 2048
	special_tokens: tuple[str, ...] = ("PAD", "BOS", "EOS", "MASK")
	use_velocities: bool = True
	use_rests: bool = False
	use_tempos: bool = False
	use_time_signatures: bool = False
	use_chords: bool = False
	use_programs: bool = False
	use_sustain_pedals: bool = False
	remove_duplicated_notes: bool = True
	default_note_duration: float = 0.5


DEFAULT_TASK3_TOKEN_CONFIG = Task3TokenConfig()


def load_maestro_split(csv_path: Path = MAESTRO_CSV) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	required_cols = {"split", "midi_filename", "canonical_composer"}
	missing = required_cols - set(df.columns)
	if missing:
		raise ValueError(f"MAESTRO CSV missing columns: {sorted(missing)}")
	return df


def build_composer_mapping(df: pd.DataFrame) -> dict[str, int]:
	composers = sorted(str(c) for c in df["canonical_composer"].dropna().unique().tolist())
	return {composer: idx for idx, composer in enumerate(composers)}


def build_remi_tokenizer(config: Task3TokenConfig = DEFAULT_TASK3_TOKEN_CONFIG) -> REMI:
	miditok_config = TokenizerConfig(
		pitch_range=(config.pitch_low, config.pitch_high),
		beat_res={(0, 4): 8, (4, 12): 4},
		num_velocities=config.num_velocities,
		special_tokens=list(config.special_tokens),
		encode_ids_split="bar",
		use_velocities=config.use_velocities,
		use_note_duration_programs=[-1],
		use_chords=config.use_chords,
		use_rests=config.use_rests,
		use_tempos=config.use_tempos,
		use_time_signatures=config.use_time_signatures,
		use_sustain_pedals=config.use_sustain_pedals,
		use_pitch_bends=False,
		use_programs=config.use_programs,
		use_pitch_intervals=False,
		use_pitchdrum_tokens=True,
		default_note_duration=config.default_note_duration,
		remove_duplicated_notes=config.remove_duplicated_notes,
	)
	return REMI(miditok_config)


def _special_token_id(tokenizer: REMI, token_name: str) -> int:
	return int(tokenizer[f"{token_name}_None"])


def normalize_token_ids(tokenizer: REMI, token_ids: Iterable[int]) -> list[int]:
	pad_id = _special_token_id(tokenizer, "PAD")
	bos_id = _special_token_id(tokenizer, "BOS")
	eos_id = _special_token_id(tokenizer, "EOS")
	cleaned = [int(token) for token in token_ids if int(token) != pad_id]
	if not cleaned:
		return []
	cleaned = [token for token in cleaned if token not in {bos_id, eos_id}]
	return [bos_id, *cleaned, eos_id]


def encode_midi_file(tokenizer: REMI, midi_path: Path) -> list[int]:
	encoded = tokenizer.encode(midi_path)
	if isinstance(encoded, list):
		encoded = encoded[0]
	if hasattr(encoded, "ids"):
		ids = list(encoded.ids)
	else:
		ids = list(encoded)
	return normalize_token_ids(tokenizer, ids)


def chunk_token_ids(
	token_ids: list[int],
	*,
	chunk_length: int,
	stride: int,
	min_chunk_length: int,
) -> Iterator[list[int]]:
	if len(token_ids) < min_chunk_length:
		return

	if len(token_ids) <= chunk_length:
		yield token_ids
		return

	bos_id = token_ids[0]
	eos_id = token_ids[-1]
	body = token_ids[1:-1]
	max_body = max(1, chunk_length - 2)
	min_body = max(1, min_chunk_length - 2)

	start = 0
	while start < len(body):
		window = body[start : start + max_body]
		if len(window) < min_body:
			if start == 0 and len(body) + 2 >= min_chunk_length:
				yield [bos_id, *body, eos_id]
			break
			break
		yield [bos_id, *window, eos_id]
		if start + max_body >= len(body):
			break
		start += stride


def _iter_split_rows(df: pd.DataFrame, split: str, maestro_root: Path) -> Iterator[tuple[Path, str, int]]:
	split_df = df[df["split"] == split]
	for _, row in split_df.iterrows():
		rel = str(row["midi_filename"])
		composer = str(row["canonical_composer"])
		yield maestro_root / rel, composer, int(row["year"])


def build_task3_token_dataset(
	*,
	maestro_root: Path = MAESTRO_ROOT,
	maestro_csv: Path = MAESTRO_CSV,
	out_dir: Path = TASK3.token_dir,
	config: Task3TokenConfig = DEFAULT_TASK3_TOKEN_CONFIG,
	limit_files: Optional[int] = None,
) -> dict[str, object]:
	"""Tokenize MAESTRO into task-3 training JSONL files."""

	df = load_maestro_split(maestro_csv)
	composer_to_id = build_composer_mapping(df)
	tokenizer = build_remi_tokenizer(config)
	out_dir.mkdir(parents=True, exist_ok=True)

	stats: dict[str, object] = {
		"maestro_root": str(maestro_root),
		"maestro_csv": str(maestro_csv),
		"token_config": asdict(config),
		"composer_to_id": composer_to_id,
		"splits": {},
	}

	for split in ("train", "validation", "test"):
		jsonl_path = getattr(TASK3, f"{split if split != 'validation' else 'val'}_jsonl")
		jsonl_path.parent.mkdir(parents=True, exist_ok=True)
		files_seen = 0
		files_loaded = 0
		files_failed = 0
		chunks_written = 0
		lengths: list[int] = []
		with jsonl_path.open("w", encoding="utf-8") as f:
			for midi_path, composer, year in _iter_split_rows(df, split, maestro_root):
				files_seen += 1
				if limit_files is not None and files_seen > limit_files:
					break
				if files_seen % 50 == 0:
					print(f"  tokenize: split={split} seen={files_seen} chunks={chunks_written}")
				if not midi_path.exists():
					files_failed += 1
					continue
				try:
					ids = encode_midi_file(tokenizer, midi_path)
				except Exception:
					files_failed += 1
					continue
				files_loaded += 1
				genre_id = int(composer_to_id.get(composer, 0))
				for chunk in chunk_token_ids(
					ids,
					chunk_length=config.chunk_length,
					stride=config.stride,
					min_chunk_length=config.min_chunk_length,
				):
					record = {
						"midi_filename": str(midi_path.relative_to(maestro_root)),
						"canonical_composer": composer,
						"year": year,
						"genre_id": genre_id,
						"token_ids": chunk,
					}
					f.write(json.dumps(record) + "\n")
					chunks_written += 1
					lengths.append(len(chunk))
		stats["splits"][split] = {
			"jsonl_path": str(jsonl_path),
			"files_seen": files_seen,
			"files_loaded": files_loaded,
			"files_failed": files_failed,
			"chunks_written": chunks_written,
			"avg_length": float(np.mean(lengths)) if lengths else 0.0,
			"max_length": int(max(lengths)) if lengths else 0,
		}

	metadata = {
		"token_config": asdict(config),
		"composer_to_id": composer_to_id,
		"pad_id": _special_token_id(tokenizer, "PAD"),
		"bos_id": _special_token_id(tokenizer, "BOS"),
		"eos_id": _special_token_id(tokenizer, "EOS"),
		"vocab_size": int(tokenizer.vocab_size),
	}
	TASK3.metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
	TASK3.tokenizer_config_json.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
	return stats


class Task3TokenDataset(Dataset):
	def __init__(self, jsonl_path: Path):
		self.jsonl_path = jsonl_path
		self.records: list[dict[str, object]] = []
		with jsonl_path.open("r", encoding="utf-8") as f:
			for line in f:
				line = line.strip()
				if not line:
					continue
				self.records.append(json.loads(line))
		if not self.records:
			raise ValueError(f"No token sequences found in {jsonl_path}")

	def __len__(self) -> int:
		return len(self.records)

	def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
		record = self.records[idx]
		return {
			"token_ids": torch.tensor(record["token_ids"], dtype=torch.long),
			"genre_id": torch.tensor(int(record["genre_id"]), dtype=torch.long),
			"length": torch.tensor(len(record["token_ids"]), dtype=torch.long),
		}


def task3_collate_batch(batch: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
	lengths = torch.tensor([int(item["length"]) for item in batch], dtype=torch.long)
	max_len = int(lengths.max().item())
	input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
	attention_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
	genre_ids = torch.stack([item["genre_id"] for item in batch], dim=0)

	for row_idx, item in enumerate(batch):
		seq = item["token_ids"]
		seq_len = int(seq.shape[0])
		input_ids[row_idx, :seq_len] = seq
		attention_mask[row_idx, :seq_len] = True

	return {
		"input_ids": input_ids,
		"attention_mask": attention_mask,
		"genre_ids": genre_ids,
		"lengths": lengths,
	}


def load_task3_metadata(metadata_json: Path = TASK3.metadata_json) -> dict[str, object]:
	return json.loads(metadata_json.read_text(encoding="utf-8"))


def load_task3_tokenizer(config_json: Path = TASK3.tokenizer_config_json) -> REMI:
	config_dict = json.loads(config_json.read_text(encoding="utf-8"))
	config = Task3TokenConfig(**config_dict)
	return build_remi_tokenizer(config)


def main() -> None:
	parser = argparse.ArgumentParser(description="Build Task 3 token dataset from MAESTRO")
	parser.add_argument("--maestro_root", type=Path, default=MAESTRO_ROOT)
	parser.add_argument("--maestro_csv", type=Path, default=MAESTRO_CSV)
	parser.add_argument("--out_dir", type=Path, default=TASK3.token_dir)
	parser.add_argument("--limit_files", type=int, default=None)
	args = parser.parse_args()
	stats = build_task3_token_dataset(
		maestro_root=args.maestro_root,
		maestro_csv=args.maestro_csv,
		out_dir=args.out_dir,
		limit_files=args.limit_files,
	)
	print(json.dumps(stats, indent=2))


if __name__ == "__main__":
	main()
