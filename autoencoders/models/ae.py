from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseAE


class AutoEncoder(BaseAE):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dim),
        )

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
