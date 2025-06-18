from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseAE


class VAE(BaseAE):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list[int],
        dropout: list[float],
        recon_weight: float = 1.0,
        kl_weight: float = 1.0,
        **kwargs
    ):
        super().__init__()
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            p = dropout[i] if i < len(dropout) else dropout[0]
            encoder_layers.append(nn.Dropout(p=p))
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for i, h_dim in enumerate(reversed(hidden_dims)):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            p = dropout[i] if i < len(dropout) else 0
            decoder_layers.append(nn.Dropout(p=p))
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        self._mu = mu  # store for loss
        self._logvar = logvar
        return self.decode(z)


    def loss(self, batch):
        x, *_ = batch
        x̂ = self.forward(x)  # returns only reconstruction

        recon_loss = F.mse_loss(x̂, x, reduction='mean')
        kl_div = -0.5 * torch.sum(
            1 + self._logvar - self._mu.pow(2) - self._logvar.exp()
        ) / x.size(0)

        total_loss = self.recon_weight * recon_loss + self.kl_weight * kl_div
        total_loss_non_weighted = recon_loss + kl_div

        return total_loss, {
            "loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "kl_div": kl_div.item(),
            "recon_weight": self.recon_weight,
            "kl_weight": self.kl_weight,
            "total_loss_non_weighted": total_loss_non_weighted.item()
        }

