"""
Fast-setup entry-point.
• builds WeightDataset right here
• performs a deterministic train/val/test split
• stores the split indices in <ckpt_dir>/splits.npz so every
  later evaluation run can reload **exactly** the same test set
• launches the generic Trainer

Run examples
------------
# train auto-encoder
python -m train model=ae run_name=ae_demo

# override split ratios
python -m train dataset.val_split=0.05 dataset.test_split=0.15

Assumes:
• configs/default.yaml is the root config
• WeightDataset is in data/weight_dataset.py
• build_model(cfg) is defined in models/registry.py
"""


from __future__ import annotations


import os
os.environ["HYDRA_FULL_ERROR"] = "1"


from pathlib import Path
from types import SimpleNamespace

import hydra
from hydra.utils import instantiate

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset, random_split

from engine.trainer import Trainer
from utils.logger import init_wandb

import sys

# add the parent directory to the path
# this is needed to import the dataset module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset import WeightDataset


# ----------------------------- helpers -------------------------------- #
def _build_splits(
    full_ds: torch.utils.data.Dataset,
    cfg: DictConfig,
    split_file: Path,
) -> tuple[Subset, Subset | None, Subset | None]:
    """
    Return (train_set, val_set, test_set).

    * If `split_file` exists → reload index arrays and build Subsets.
    * Else  → draw deterministic split, save indices, and return Subsets.
    """
    if split_file.exists():
        idx = np.load(split_file)
        train_idx, val_idx, test_idx = idx["train"], idx["val"], idx["test"]
        return (
            Subset(full_ds, train_idx),
            Subset(full_ds, val_idx) if len(val_idx) else None,
            Subset(full_ds, test_idx) if len(test_idx) else None,
        )

    # ---------- first time: create the split deterministically ----------
    total_len = len(full_ds)
    val_len = int(total_len * cfg.dataset.val_split)
    test_len = int(total_len * cfg.dataset.test_split)
    train_len = total_len - val_len - test_len

    g = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set, test_set = random_split(
        full_ds, [train_len, val_len, test_len], generator=g
    )

    # stash the integer index arrays for reproducibility
    np.savez(
        split_file,
        train=np.array(train_set.indices, dtype=np.int64),
        val=np.array(val_set.indices, dtype=np.int64),
        test=np.array(test_set.indices, dtype=np.int64),
    )
    return train_set, val_set, test_set



# ---------------------------------------------------------------------- #
@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    # 1.  initialise wandb
    run = init_wandb(cfg, run_name=cfg.run_name)

    # 2.  build the full WeightDataset
    full_ds = WeightDataset(
        mlps_folder=cfg.dataset.root,
        wandb_logger=None,  # trainer handles logging
        model_dims=cfg.dataset.model_dims,
        mlp_kwargs=cfg.dataset.mlp_kwargs,
        cfg=cfg.dataset,
    )

    # 3.  split & (first time) save indices
    ckpt_dir = Path(cfg.trainer.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    split_file = ckpt_dir / "splits.npz"

    train_set, val_set, test_set = _build_splits(full_ds, cfg, split_file)

    # 4.  wrap splits in a tiny "datamodule" the Trainer expects
    datamodule = SimpleNamespace(train=train_set, val=val_set, test=test_set)

    # 5.  instantiate the model from YAML (_target_ field)
    model = instantiate(cfg.model)

    # 6.  launch training
    trainer = Trainer(model, datamodule, cfg, run_name=run.name)
    trainer.fit()

    run.finish()


if __name__ == "__main__":
    main()
