import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseAE


class AutoEncoder(BaseAE):
    def __init__(self, latent_dim: int, hidden_dims: list[int]):
        super().__init__()
        # very small example encoder / decoder
        self.encoder = nn.Sequential(
            nn.Linear(1_024, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], 1_024),
        )

    # ----------------------------------------- #
    def encode(self, x):          # x: (B, 1024)
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def loss(self, batch):
        x, *_ = batch                         # ignore prev_weights
        x̂ = self.forward(x)
        recon_loss = F.mse_loss(x̂, x)
        return recon_loss, {"recon": recon_loss.item()}
