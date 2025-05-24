from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseAE


class AutoEncoder(BaseAE):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list[int]):
        super().__init__()

        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder (reverse hidden_dims)
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    # ----------------------------------------- #
    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def loss(self, batch):
        x, *_ = batch
        x̂ = self.forward(x)
        recon_loss = F.mse_loss(x̂, x)
        return recon_loss, {"recon": recon_loss.item()}
