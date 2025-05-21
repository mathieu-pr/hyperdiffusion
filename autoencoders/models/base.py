from __future__ import annotations
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseAE(nn.Module, ABC):
    """
    Shared contract:
      * forward(x) -> x̂ (reconstruction)
      * loss(batch) -> total_loss, dict_of_named_components
      * sample(n)   -> generated samples (optional, needed for FID)
    """

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        ...

    def forward(self, x):
        return self.decode(self.encode(x))

    @abstractmethod
    def loss(self, batch) -> tuple[torch.Tensor, dict]:
        ...

    # ───────────────────── optional helpers ────────────────────── #
    def sample(self, n: int) -> torch.Tensor:
        raise NotImplementedError
