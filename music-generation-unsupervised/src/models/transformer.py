"""Decoder-only Transformer model (Task 3)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TransformerHParams:
	vocab_size: int
	genre_vocab_size: int
	d_model: int = 256
	nhead: int = 8
	num_layers: int = 4
	dim_feedforward: int = 1024
	dropout: float = 0.1
	max_seq_len: int = 2048
	pad_token_id: int = 0
	bos_token_id: int = 1
	eos_token_id: int = 2


class DecoderOnlyTransformer(nn.Module):
	def __init__(self, hparams: TransformerHParams):
		super().__init__()
		if hparams.d_model % hparams.nhead != 0:
			raise ValueError("d_model must be divisible by nhead")
		self.hparams = hparams
		self.token_embedding = nn.Embedding(
			hparams.vocab_size,
			hparams.d_model,
			padding_idx=hparams.pad_token_id,
		)
		self.genre_embedding = nn.Embedding(hparams.genre_vocab_size, hparams.d_model)
		self.position_embedding = nn.Embedding(hparams.max_seq_len, hparams.d_model)
		self.input_dropout = nn.Dropout(hparams.dropout)
		self.input_norm = nn.LayerNorm(hparams.d_model)
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=hparams.d_model,
			nhead=hparams.nhead,
			dim_feedforward=hparams.dim_feedforward,
			dropout=hparams.dropout,
			batch_first=True,
			activation="gelu",
		)
		self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=hparams.num_layers)
		self.output_norm = nn.LayerNorm(hparams.d_model)
		self.output_head = nn.Linear(hparams.d_model, hparams.vocab_size)

	@staticmethod
	def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
		mask = torch.triu(
			torch.ones((seq_len, seq_len), device=device, dtype=torch.bool),
			diagonal=1,
		)
		return mask

	def forward(
		self,
		input_ids: torch.Tensor,
		*,
		genre_ids: torch.Tensor | None = None,
		attention_mask: torch.Tensor | None = None,
	) -> torch.Tensor:
		batch_size, seq_len = input_ids.shape
		if seq_len > self.hparams.max_seq_len:
			raise ValueError(
				f"Sequence length {seq_len} exceeds max_seq_len={self.hparams.max_seq_len}"
			)

		positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
		token_emb = self.token_embedding(input_ids)
		pos_emb = self.position_embedding(positions)
		if genre_ids is None:
			genre_ids = torch.zeros(batch_size, dtype=torch.long, device=input_ids.device)
		genre_emb = self.genre_embedding(genre_ids).unsqueeze(1)
		x = token_emb + pos_emb + genre_emb
		x = self.input_norm(self.input_dropout(x))

		causal_mask = self._causal_mask(seq_len, input_ids.device)
		key_padding_mask = None
		if attention_mask is not None:
			key_padding_mask = ~attention_mask.bool()
		out = self.transformer(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
		out = self.output_norm(out)
		return self.output_head(out)

	@torch.no_grad()
	def generate(
		self,
		start_tokens: torch.Tensor,
		*,
		genre_ids: torch.Tensor,
		max_new_tokens: int,
		temperature: float = 1.0,
		top_k: int = 0,
	) -> torch.Tensor:
		self.eval()
		if start_tokens.ndim == 1:
			start_tokens = start_tokens.unsqueeze(0)
		generated = start_tokens.clone()
		if genre_ids.ndim == 0:
			genre_ids = genre_ids.unsqueeze(0)

		for _ in range(max_new_tokens):
			context = generated[:, -self.hparams.max_seq_len :]
			attention_mask = context != self.hparams.pad_token_id
			logits = self.forward(context, genre_ids=genre_ids, attention_mask=attention_mask)
			next_logits = logits[:, -1, :] / max(temperature, 1e-6)
			if top_k > 0:
				values, indices = torch.topk(next_logits, k=min(top_k, next_logits.shape[-1]), dim=-1)
				filtered = torch.full_like(next_logits, float("-inf"))
				filtered.scatter_(dim=-1, index=indices, src=values)
				next_logits = filtered
			probs = torch.softmax(next_logits, dim=-1)
			next_token = torch.multinomial(probs, num_samples=1)
			generated = torch.cat([generated, next_token], dim=1)
			if torch.all(next_token.squeeze(-1) == self.hparams.eos_token_id):
				break
		return generated
