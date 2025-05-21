import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseAE


class VariationalAutoEncoder(BaseAE):
    def __init__(self, latent_dim: int, hidden_dims: list[int], beta: float = 1.0):
        super().__init__()
        self.beta = beta
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        # …build encoder, decoder…

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(self, batch):
        x, *_ = batch
        x̂, mu, logvar = self.forward(x)
        recon = F.mse_loss(x̂, x, reduction="mean")
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = recon + self.beta * kld
        return total, {"recon": recon.item(), "kld": kld.item()}
