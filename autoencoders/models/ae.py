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
        # print("intial x we are trying to encode:", x)
        # print(f"Encoding input of shape {x.shape} to latent space")
        encoded = self.encoder(x)
        # print(f"self.encoder(x) shape: {encoded.shape}")
        # print(f"self.encoder(x): {encoded}")
        return encoded
    

        # print("\n\ninital x shape:", x.shape)

        # # Print the 15 greatest values of each vector in x and the mean of the corresponding features
        # topk = 15
        # values, indices = torch.topk(x, topk, dim=1)
        # print(f"Top {topk} values per vector in x:\n{values}")
        # print(f"Indices of top {topk} values per vector in x:\n{indices}")
        # # For each feature index in the topk, print the mean of that feature across the batch
        # for i in range(x.shape[0]):
        #     print(f"Vector {i}:")
        #     for j in range(topk):
        #         feature_idx = indices[i, j].item()
        #         feature_mean = x[:, feature_idx].mean().item()
        #         print(f"  Feature {feature_idx}: value={values[i, j].item():.4f}, mean_across_batch={feature_mean:.4f}")

        # print("inital x:", x)
        # print("initial x mean/std:", x.mean().item(), x.std().item())
        # for i, layer in enumerate(self.encoder):
        #     x = layer(x)
        #     print(f"After layer {i} ({layer.__class__.__name__}): mean={x.mean().item():.4f}, std={x.std().item():.4f}, has_nan={torch.isnan(x).any().item()}")
        # return x

    def decode(self, z):
        # print(f"Decoding latent vector of shape {z.shape} to input space")
        decoded = self.decoder(z)
        # print(f"self.decoder(z) shape: {decoded.shape}")
        # print(f"self.decoder(z): {decoded}")
        return decoded

    def loss(self, batch):
        x, *_ = batch
        x̂ = self.forward(x)
        recon_loss = F.mse_loss(x̂, x)
        return recon_loss, {"recon": recon_loss.item()}
