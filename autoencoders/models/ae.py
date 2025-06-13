from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseAE


class AutoEncoder(BaseAE):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list[int], dropout: list[float]):
        super().__init__()

        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            # Apply dropout if available, otherwise use 0
            p = dropout[i] if i < len(dropout) else dropout[0]
            encoder_layers.append(nn.Dropout(p=p))
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder (reverse hidden_dims)
        decoder_layers = []
        prev_dim = latent_dim
        reversed_hidden_dims = list(reversed(hidden_dims))
        for i, h_dim in enumerate(reversed_hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            # Apply dropout if available, otherwise use 0
            p = dropout[i] if i < len(dropout) else 0
            decoder_layers.append(nn.Dropout(p=p))
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    # ----------------------------------------- #
    def encode(self, x):
        encoded = self.encoder(x)
        return encoded

    def decode(self, z):
        decoded = self.decoder(z)
        return decoded

    def loss(self, batch):
        x, *_ = batch
        x̂ = self.forward(x)
        recon_loss = F.mse_loss(x̂, x)
        return recon_loss, {"recon": recon_loss.item()}
