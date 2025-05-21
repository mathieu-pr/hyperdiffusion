import torch
import torch.nn.functional as F


def psnr(x̂, x, max_val: float = 1.0):
    mse = F.mse_loss(x̂, x)
    return 20 * torch.log10(max_val) - 10 * torch.log10(mse)