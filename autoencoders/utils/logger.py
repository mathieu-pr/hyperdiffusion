"""
All wandb-related helpers live here so the rest of the code never
imports wandb directly.
"""
from __future__ import annotations
import os
import wandb
from omegaconf import OmegaConf


def init_wandb(cfg, run_name: str):
    """Initialise a wandb run and return the Run handle."""
    os.environ.setdefault("WANDB_SILENT", "true")   # keep Colab output clean

    run = wandb.init(
        project=cfg.logging.project,
        name=run_name,
        dir=cfg.logging.log_dir,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        resume="allow",
        save_code=True,
    )
    return run


def log_metrics(step: int | None = None, **kwargs):
    """
    Thin wrapper so you can write:
        log_metrics(step=global_step, loss=loss, psnr=psnr)
    from anywhere.
    """
    if wandb.run:
        wandb.log(kwargs, step=step)
    else:
        print("wandb not initialised, skipping log_metrics")
