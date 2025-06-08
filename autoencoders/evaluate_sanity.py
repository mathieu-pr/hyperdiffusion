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

import os
os.environ["HYDRA_FULL_ERROR"] = "1"

from pathlib import Path
from types import SimpleNamespace

import hydra
from hydra.utils import instantiate
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Subset

from engine.evaluator import Evaluator
from utils.logger import init_wandb

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset import WeightDataset
from hyperdiffusion import HyperDiffusion
from hd_utils import Config
import ldm.ldm.modules.diffusionmodules.openaimodel
import wandb
from transformer import Transformer
from hd_utils import Config, get_mlp
import glob


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

def get_model_checkpoint(folder_path):
    # Look for best_model_run_vl*.pt
    best_model_pattern = os.path.join(folder_path, "best_model_run_vl*.pt")
    best_models = glob.glob(best_model_pattern)
    
    if best_models:
        return best_models[0]  # Return the first match
    
    # Fallback to last_epoch*.pt
    last_epoch_pattern = os.path.join(folder_path, "last_epoch*.pt")
    last_epochs = glob.glob(last_epoch_pattern)
    
    if last_epochs:
        return last_epochs[0]
    
    # If neither exists
    return None


# -------------------------------------------------------------------- #
@hydra.main(version_base=None, config_path="configs", config_name="default_eval")
def main(cfg: DictConfig):
    # 1) initialise wandb for the evaluation run
    
    run = init_wandb(cfg, run_name=f"eval_{Path(cfg.ckpt_path).parent.name}")

    normalization_stats_path = Path(cfg.normalization_path)
    normalization_stats = torch.load(normalization_stats_path)

    # 2) rebuild the full dataset exactly as in training
    full_ds = WeightDataset(
        mlps_folder=cfg.dataset.root,
        wandb_logger=None,
        model_dims=cfg.dataset.model_dims,
        mlp_kwargs=cfg.dataset.mlp_kwargs,
        cfg=cfg.dataset,
        should_normalize=True,  # enable normalization
        normalization_stats_path=normalization_stats_path
    )

    # 3) restore the index split saved during training
    split_file = Path(cfg.split_path)
    train_set, val_set, test_set = _load_splits(full_ds, split_file)

    train_set = Subset(train_set, range(4))
    splits = SimpleNamespace(train=train_set, val=val_set, test=test_set)

    # 4) rebuild the AE model from YAML + load weights
    model_AE = instantiate(cfg.model)
    print(cfg.ckpt_path)
    ckpt_path = get_model_checkpoint(cfg.ckpt_path)
    state_dict = torch.load(ckpt_path, map_location=cfg.device)
    model_AE.load_state_dict(state_dict)

    # 5a) make a hyperdiffusion instance
    Config.config = config = cfg
    method = cfg.train_plane.method
    mlp_kwargs = None

    # In HyperDiffusion, we need to know the specifications of MLPs that are used for overfitting
    if "hyper" in method:
        mlp_kwargs = Config.config.train_plane["mlp_config"]["params"]

    # Initialize Transformer for HyperDiffusion
    if "hyper" in method:
        mlp = get_mlp(mlp_kwargs)
        state_dict = mlp.state_dict()
        layers = []
        layer_names = []
        for l in state_dict:
            shape = state_dict[l].shape
            layers.append(np.prod(shape))
            layer_names.append(l)
        model = Transformer(
            layers, layer_names, **Config.config.train_plane["transformer_config"]["params"]
        ).to(cfg.device).float()
    # Initialize UNet for Voxel baseline
    else:
        model = ldm.ldm.modules.diffusionmodules.openaimodel.UNetModel(
            **Config.config.train_plane["unet_config"]["params"]
        ).float()

    # Infer image_shape from first sample (assuming dataset returns tensors)
    first_sample = train_set[0][0] if train_set is not None else val_set[0][0]
    image_shape = (cfg.eval.batch_size, ) + tuple(first_sample.shape)  # e.g. (B, C)

    # Create HyperDiffusion instance
    hyperdiffusion_instance = HyperDiffusion(
        model=model,
        train_dt=train_set,
        val_dt=val_set,
        test_dt=test_set,
        mlp_kwargs=mlp_kwargs,
        image_shape=image_shape,
        method="hyper_3d",
        cfg=cfg
    )

    # 5) run evaluator (prefers .test, falls back to .val)
    evaluator = Evaluator(model_AE, splits, cfg, run_dir=Path(cfg.ckpt_path), split="train", hyperdiffusion_obj=hyperdiffusion_instance, normalization_stats_path=normalization_stats_path) #### model is AE model
    evaluator.run()        # cfg.eval.split usually "test"

    run.finish()

# IN ORDER TO RUN, CHANGE THE CKPT_PATH AND THEN CHANGE THE SANITY SIZE

if __name__ == "__main__":
    main()
