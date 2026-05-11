"""LSTM Autoencoder model (Task 1).

Input: piano-roll window X of shape (B, T, 88) with binary values in {0, 1}.
Output: reconstruction logits of shape (B, T, 88).

We return logits (not probabilities) so training can use numerically stable
losses like BCEWithLogitsLoss / focal loss with logits.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class AutoencoderHParams:
	input_dim: int = 88
	hidden_dim: int = 256
	latent_dim: int = 64
	num_layers: int = 2
	dropout: float = 0.2


class LSTMAutoencoder(nn.Module):
	def __init__(self, hparams: AutoencoderHParams):
		super().__init__()
		self.hparams = hparams

		enc_dropout = hparams.dropout if hparams.num_layers > 1 else 0.0
		self.encoder = nn.LSTM(
			input_size=hparams.input_dim,
			hidden_size=hparams.hidden_dim,
			num_layers=hparams.num_layers,
			dropout=enc_dropout,
			batch_first=True,
		)
		self.to_latent = nn.Linear(hparams.hidden_dim, hparams.latent_dim)

		dec_dropout = hparams.dropout if hparams.num_layers > 1 else 0.0
		self.decoder = nn.LSTM(
			input_size=hparams.latent_dim,
			hidden_size=hparams.hidden_dim,
			num_layers=hparams.num_layers,
			dropout=dec_dropout,
			batch_first=True,
		)
		self.to_logits = nn.Linear(hparams.hidden_dim, hparams.input_dim)

	def encode(self, x: torch.Tensor) -> torch.Tensor:
		"""Encode x -> z (B, latent_dim)."""
		# hn: (num_layers, B, hidden_dim)
		_, (hn, _) = self.encoder(x)
		last = hn[-1]
		z = self.to_latent(last)
		return z

	def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
		"""Decode z -> logits (B, T, 88)."""
		# Repeat z across time so the decoder sees the latent code at every step.
		z_seq = z.unsqueeze(1).repeat(1, seq_len, 1)
		out, _ = self.decoder(z_seq)
		logits = self.to_logits(out)
		return logits

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		z = self.encode(x)
		return self.decode(z, seq_len=x.shape[1])
