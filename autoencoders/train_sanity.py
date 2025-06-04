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
from datetime import datetime


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
    # if split_file.exists():
    #     print(f"\n\nloading existing split from {split_file}")
    #     idx = np.load(split_file)
    #     train_idx, val_idx, test_idx = idx["train"], idx["val"], idx["test"]

    #     print(f"train_len: {len(train_idx)}, val_len: {len(val_idx)}, test_len: {len(test_idx)}")
        
    #     return (
    #         Subset(full_ds, train_idx),
    #         Subset(full_ds, val_idx) if len(val_idx) else None,
    #         Subset(full_ds, test_idx) if len(test_idx) else None,
    #     )

    # ---------- first time: create the split deterministically ----------
    total_len = len(full_ds)
    val_len = int(total_len * cfg.dataset.val_split)
    test_len = int(total_len * cfg.dataset.test_split)
    train_len = total_len - val_len - test_len

    # create a descriptive split file name with date, seed, and split ratios
    date_str = datetime.now().strftime("%Y%m%d")
    split_file_with_info = split_file.parent / (
        f"splits_seed{cfg.seed}_totallen{total_len}_val{cfg.dataset.val_split}_test{cfg.dataset.test_split}.npz"
    )


    print(f"\n\ncreating new split: {split_file_with_info}")
    print(f"train_len: {train_len}, val_len: {val_len}, test_len: {test_len}\n\n")

    g = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set, test_set = random_split(
        full_ds, [train_len, val_len, test_len], generator=g
    )

    
    # stash the integer index arrays for reproducibility
    np.savez(
        split_file_with_info,
        train=np.array(train_set.indices, dtype=np.int64),
        val=np.array(val_set.indices, dtype=np.int64),
        test=np.array(test_set.indices, dtype=np.int64),
    )
    return train_set, val_set, test_set





# ---------------------------------------------------------------------- #
@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    # 1.  Initialise wandb
    run = init_wandb(cfg, run_name=cfg.run_name)

    print(f"\n\n{run.name} - {cfg.trainer.model_name}\n")
    print(f"cfg.model: {cfg.model}\n")
    print(f"\ncfg.dataset.root: {cfg.dataset.root}\n")

    # 2.  Build full dataset WITHOUT normalization (to compute mean/std)
    full_ds = WeightDataset(
        mlps_folder=cfg.dataset.root,
        wandb_logger=None,
        model_dims=cfg.dataset.model_dims,
        mlp_kwargs=cfg.dataset.mlp_kwargs,
        cfg=cfg.dataset,
        should_normalize=False,  # disable normalization for now
        normalization_stats_path=None  # disable normalization for now
    )

    # 3.  Split & save indices
    ckpt_dir = Path(cfg.trainer.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    split_file = ckpt_dir / "splits.npz"

    print(f"\nlen(full_ds): {len(full_ds)}")
    train_set, val_set, test_set = _build_splits(full_ds, cfg, split_file)

    # 4.  Compute normalization stats on training set only
    threshold_std = 1e-5 

    normalization_stats_path = Path(cfg.trainer.ckpt_dir) / f"normalization_stats_totallen{len(full_ds)}_val{cfg.dataset.val_split}_test{cfg.dataset.test_split}_thresholdSTD_{threshold_std}.pt"
    print(f"\n\nnormalization_stats_path: {normalization_stats_path}\n")

    if not normalization_stats_path.exists():
        print("→ Computing normalization stats from training set only...")
        all_train_weights = []
        for idx in train_set.indices:
            weights, *_ = full_ds[idx]  # full_ds is unnormalized
            all_train_weights.append(weights)
        all_train_weights = torch.stack(all_train_weights)

        #print the number of samples used to compute normalization stats
        print(f"→ Number of samples used for normalization stats: {len(all_train_weights)}")

        mean = all_train_weights.mean(dim=0)
        std = all_train_weights.std(dim=0)

        #replace NaN values in mean and std with 0 and 1 respectively
        std = torch.where(torch.isnan(std), torch.ones_like(std), std)

        #prevent problem when std is very small in training set and not in validation/test sets
        std = std.clamp(min=threshold_std)

        torch.save({"mean": mean, "std": std}, normalization_stats_path)
        print(f"→ Saved normalization stats to {normalization_stats_path}")
    else:
        print(f"→ Using existing normalization stats: {normalization_stats_path}")

    #print normalization stats
    normalization_stats = torch.load(normalization_stats_path)
    print(f"\nNormalization stats: mean={normalization_stats['mean']}, std={normalization_stats['std']}\n")

    #print the shape of the mean and std
    print(f"mean shape: {normalization_stats['mean'].shape}, std shape: {normalization_stats['std'].shape}\n")



    # 5.  Rebuild dataset WITH normalization
    norm_ds = WeightDataset(
        mlps_folder=cfg.dataset.root,
        wandb_logger=None,
        model_dims=cfg.dataset.model_dims,
        mlp_kwargs=cfg.dataset.mlp_kwargs,
        cfg=cfg.dataset,
        should_normalize=True,  # enable normalization
        normalization_stats_path=normalization_stats_path
    )


   

    # Redo the splits
    train_set, val_set, test_set = _build_splits(norm_ds, cfg, split_file)
    print(f"\n\ntrain_set: {len(train_set)}, val_set: {len(val_set) if val_set else 'None'}, test_set: {len(test_set) if test_set else 'None'}\n")


    ######## Sanity check #######

    #Number of training samples is set to 1 for sanity check
    number_training_samples = 1
    train_set = Subset(train_set, range(number_training_samples))
    #or train_set = train_set[:number_training_samples]
    print(f"Using {number_training_samples} training samples for sanity check\n")

    # 6. Wrap in datamodule
    datamodule = SimpleNamespace(train=train_set, val=val_set, test=test_set)

    # 7. Instantiate model
    model = instantiate(cfg.model)

    # 8. Launch training
    trainer = Trainer(model, datamodule, cfg, run_name=run.name, normalization_stats_path=normalization_stats_path)
    trainer.fit()

    # 9. Finish wandb
    run.finish()


if __name__ == "__main__":
    main()
