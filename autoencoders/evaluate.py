# autoencoders/evaluate.py
"""
Offline evaluation script.

Usage
-----
python -m evaluate ckpt_path=<path_to_best.pt>

• Re-creates the SAME train/val/test splits by loading the
  <ckpt_dir>/splits.npz file saved during training.
• Runs engine.evaluator.Evaluator on the test split (falls
  back to val if you set test_split: 0).
• Logs metrics to wandb and writes eval.json next to the ckpt.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Subset

from data.weight_dataset import WeightDataset
from engine.evaluator import Evaluator
from utils.logger import init_wandb


# -------------------------------------------------------------------- #
def _load_splits(full_ds, split_file: Path):
    """
    Returns (train_set, val_set, test_set) Subsets using the index arrays
    stored in split_file.  If the file is missing, raises FileNotFoundError.
    """
    if not split_file.exists():
        raise FileNotFoundError(
            f"Cannot find {split_file}.  "
            "Run training first so the split indices are stored."
        )
    idx = np.load(split_file)
    train_idx, val_idx, test_idx = idx["train"], idx["val"], idx["test"]
    train_set = Subset(full_ds, train_idx)
    val_set   = Subset(full_ds, val_idx)   if len(val_idx)   else None
    test_set  = Subset(full_ds, test_idx)  if len(test_idx)  else None
    return train_set, val_set, test_set


# -------------------------------------------------------------------- #
@hydra.main(config_path="configs", config_name="default")
def main(cfg: DictConfig):
    # 1) initialise wandb for the evaluation run
    run = init_wandb(cfg, run_name=f"eval_{Path(cfg.ckpt_path).parent.name}")

    # 2) rebuild the full dataset exactly as in training
    full_ds = WeightDataset(
        mlps_folder=cfg.dataset.root,
        wandb_logger=None,
        model_dims=cfg.dataset.model_dims,
        mlp_kwargs=cfg.dataset.mlp_kwargs,
        cfg=cfg.dataset,
    )

    # 3) restore the index split saved during training
    split_file = Path(cfg.ckpt_path).parent / "splits.npz"
    train_set, val_set, test_set = _load_splits(full_ds, split_file)

    splits = SimpleNamespace(train=train_set, val=val_set, test=test_set)

    # 4) rebuild the model from YAML + load weights
    model = build_model(cfg)
    state_dict = torch.load(cfg.ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)

    # 5) run evaluator (prefers .test, falls back to .val)
    evaluator = Evaluator(model, splits, cfg, run_dir=Path(cfg.ckpt_path).parent)
    evaluator.run(split=cfg.eval.split)        # cfg.eval.split usually "test"

    run.finish()


if __name__ == "__main__":
    main()
