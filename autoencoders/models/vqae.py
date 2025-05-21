from .base import BaseAE

class VQVAE(BaseAE):
    def __init__(self, latent_dim: int, codebook_size: int, **kwargs):
        super().__init__()
        # build encoder, codebook, decoder …

    def encode(self, x):
        ...

    def decode(self, z):
        ...

    def loss(self, batch):
        # return total, {"recon": …, "commit": …, "perplex": …}
        ...
