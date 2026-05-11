"""LSTM Variational Autoencoder model (Task 2)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class VAEHParams:
	input_dim: int = 88
	hidden_dim: int = 256
	latent_dim: int = 64
	num_layers: int = 2
	dropout: float = 0.2


class LSTMVAE(nn.Module):
	def __init__(self, hparams: VAEHParams):
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
		self.to_mu = nn.Linear(hparams.hidden_dim, hparams.latent_dim)
		self.to_logvar = nn.Linear(hparams.hidden_dim, hparams.latent_dim)

		dec_dropout = hparams.dropout if hparams.num_layers > 1 else 0.0
		self.decoder = nn.LSTM(
			input_size=hparams.latent_dim,
			hidden_size=hparams.hidden_dim,
			num_layers=hparams.num_layers,
			dropout=dec_dropout,
			batch_first=True,
		)
		self.to_logits = nn.Linear(hparams.hidden_dim, hparams.input_dim)

	def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		"""Encode x -> (mu, logvar)."""
		_, (hn, _) = self.encoder(x)
		last = hn[-1]
		mu = self.to_mu(last)
		logvar = self.to_logvar(last)
		return mu, logvar

	def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
		"""Sample z using the reparameterization trick."""
		if self.training:
			std = torch.exp(0.5 * logvar)
			eps = torch.randn_like(std)
			return mu + eps * std
		return mu

	def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
		z_seq = z.unsqueeze(1).repeat(1, seq_len, 1)
		out, _ = self.decoder(z_seq)
		return self.to_logits(out)

	def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		logits = self.decode(z, seq_len=x.shape[1])
		return logits, mu, logvar

	def sample(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
		z = torch.randn(batch_size, self.hparams.latent_dim, device=device)
		return self.decode(z, seq_len=seq_len)
