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
import yaml

from engine.trainer import Trainer
from utils.logger import init_wandb

import sys

# add the parent directory to the path
# this is needed to import the dataset module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset import WeightDataset
from datetime import datetime

import warnings

# Suppress specific FutureWarnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*torch.load.*weights_only=False.*"
)

warnings.filterwarnings(
    "ignore",
    message=".*Metric `FrechetInceptionDistance` will save all extracted features in buffer.*",
    category=UserWarning
)


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

    print(f"creating new split: {split_file_with_info}")

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


def read_and_modify_one_block_of_yaml_data(
        filepath_origin: str,
        filepath_destination: str, 
        key: str, 
        value: any
        ):
    with open(f'{filepath_origin}', 'r') as f:
        data = yaml.safe_load(f)
        if key == 'normalization_path':
            data[f'{key}'] = data[f'{key}'] + '' + f'{value}'
        else :
            data[f'{key}'] = f'{value}'
    with open(f'{filepath_destination}', 'w') as file:
        yaml.dump(data,file,sort_keys=False)


# ---------------------------------------------------------------------- #
@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    torch.set_num_threads(2)


    print("\n" + "="*80)
    print("AUTOENCODER TRAINING INITIALIZATION")
    print("="*80)

    # 1. Initialize wandb
    print("\n1. WANDB INITIALIZATION")
    print("-"*30)
    run = init_wandb(cfg, run_name=cfg.run_name)
    print(f"Run name:        {run.name}")
    print(f"Model name:      {cfg.trainer.model_name}")
    print(f"Dataset root:    {cfg.dataset.root}")
    print("\nTraining Parameters:")
    print(f"  Learning rate:  {cfg.optimizer.lr}")
    print(f"  Weight decay:   {cfg.optimizer.wd}")
    print(f"  Batch size:     {cfg.trainer.batch_size}")
    print(f"Model config:     {cfg.model}")

    # 2. Build initial dataset
    print("\n2. DATASET INITIALIZATION")
    print("-"*30)
    print("Building initial dataset without normalization...")
    full_ds = WeightDataset(
        mlps_folder=cfg.dataset.root,
        wandb_logger=None,
        model_dims=cfg.dataset.model_dims,
        mlp_kwargs=cfg.dataset.mlp_kwargs,
        cfg=cfg.dataset,
        should_normalize=False,
        normalization_stats_path=None
    )
    print(f"Total dataset size: {len(full_ds)}")

    # 3. Split & save indices
    print("\n3. DATASET SPLITTING")
    print("-"*30)
    ckpt_dir = Path(cfg.trainer.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    split_file = ckpt_dir / "splits.npz"
    
    train_set, val_set, test_set = _build_splits(full_ds, cfg, split_file)
    print("\nSplit Information:")
    print(f"  Training set:   {len(train_set.indices)} samples")
    print(f"  Validation set: {len(val_set.indices) if val_set else 0} samples")
    print(f"  Test set:       {len(test_set.indices) if test_set else 0} samples")

    # 4. Compute normalization stats
    print("\n4. NORMALIZATION STATISTICS")
    print("-"*30)
    threshold_std = 1e-3
    normalization_stats_name = (
        f"normalization_stats_totallen{len(full_ds)}_"
        f"val{cfg.dataset.val_split}_test{cfg.dataset.test_split}_"
        f"thresholdSTD_{threshold_std}.pt"
    )
    normalization_stats_path = Path(cfg.trainer.ckpt_dir) / normalization_stats_name
    print(f"Stats file: {normalization_stats_path}")

    if not normalization_stats_path.exists():
        print("\nComputing new normalization stats...")
        all_train_weights = []
        for idx in train_set.indices:
            weights, *_ = full_ds[idx]
            all_train_weights.append(weights)
        all_train_weights = torch.stack(all_train_weights)
        
        print(f"Using {len(all_train_weights)} training samples for statistics")
        
        mean = all_train_weights.mean(dim=0)
        std = all_train_weights.std(dim=0)
        std = torch.where(torch.isnan(std), torch.ones_like(std), std)
        std = std.clamp(min=threshold_std)
        
        torch.save({"mean": mean, "std": std}, normalization_stats_path)
        print("→ Normalization stats saved successfully")
    else:
        print("\nLoading existing normalization stats...")
    
    stats = torch.load(normalization_stats_path)
    print("\nNormalization Statistics Shape:")
    print(f"  Mean shape: {stats['mean'].shape}")
    print(f"  Std shape:  {stats['std'].shape}")

    # 5. Build normalized dataset
    norm_ds = WeightDataset(
        mlps_folder=cfg.dataset.root,
        wandb_logger=None,
        model_dims=cfg.dataset.model_dims,
        mlp_kwargs=cfg.dataset.mlp_kwargs,
        cfg=cfg.dataset,
        should_normalize=True,
        normalization_stats_path=normalization_stats_path
    )

    # Redo splits with normalized data
    train_set, val_set, test_set = _build_splits(norm_ds, cfg, split_file)
    print("\nNormalized Dataset Splits:")
    print(f"  Training:   {len(train_set)} samples")
    print(f"  Validation: {len(val_set) if val_set else 0} samples")
    print(f"  Test:       {len(test_set) if test_set else 0} samples")

    # 6. Training setup
    datamodule = SimpleNamespace(train=train_set, val=val_set, test=test_set)
    model = instantiate(cfg.model)

    # 7. Launch training
    print("-"*30, "\n\n")
    trainer = Trainer(model, datamodule, cfg, run_name=run.name, 
                     normalization_stats_path=normalization_stats_path)
    ckpt_folder_path = trainer.fit()

    # 8. Update eval config
    print("\n8. UPDATING EVALUATION CONFIG")
    print("-"*30)
    read_and_modify_one_block_of_yaml_data(
        './autoencoders/configs/default.yaml',
        './autoencoders/configs/default_eval.yaml',
        'normalization_path',
        normalization_stats_name
    )
    read_and_modify_one_block_of_yaml_data(
        './autoencoders/configs/default_eval.yaml',
        './autoencoders/configs/default_eval.yaml',
        'ckpt_path',
        cfg.ckpt_path + ckpt_folder_path
    )
    print("Evaluation config updated successfully")

    
    run.finish()
    print("\nTraining completed successfully!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
