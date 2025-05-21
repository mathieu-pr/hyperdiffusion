# src/engine/trainer.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from utils.logger import log_metrics
from utils.metrics import psnr  # or any other quick-val metric you prefer


class Trainer:
    """
    Minimal, framework-free trainer that

    • trains on `datamodule.train`
    • runs a *cheap* validation pass on `datamodule.val` every epoch
    • logs everything to wandb through `utils.logger.log_metrics`
    • tracks the metric in `cfg.trainer.monitor`
      (e.g. "val/psnr"  or  "val/loss")
      and saves the state-dict of the best model to disk **and** wandb
    """

    # --------------------------------------------------------------------- #
    def __init__(self, model, datamodule, cfg, run_name: str):
        self.model = model.to(cfg.device)
        self.cfg = cfg
        self.run_name = run_name

        # ───────────── loaders ──────────────────────────────────────────── #
        self.train_loader = DataLoader(
            datamodule.train,
            batch_size=cfg.trainer.batch_size,
            shuffle=True,
            num_workers=cfg.dataset.num_workers,
            pin_memory=False,
        )

        # `val` split is optional but strongly recommended
        self.val_loader = (
            DataLoader(
                datamodule.val,
                batch_size=cfg.trainer.batch_size,
                shuffle=False,
                num_workers=cfg.dataset.num_workers,
                pin_memory=False,
            )
            if hasattr(datamodule, "val") and datamodule.val is not None
            else None
        )

        # ───────────── optimisation ─────────────────────────────────────── #
        self.opt = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.wd,
        )

        # ───────────── checkpoint bookkeeping ──────────────────────────── #
        self.ckpt_dir: Path = Path(cfg.trainer.ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_value: float | None = None
        self.best_path: Path | None = None
        self.monitor: str = cfg.trainer.monitor      # e.g. "val/psnr"
        self.mode: str = cfg.trainer.mode.lower()    # "min" or "max"
        assert self.mode in {"min", "max"}, "trainer.mode must be 'min' or 'max'"

        # convenience: split monitor name → key after "val/"
        self._monitor_key = self.monitor.split("/", 1)[1] if "/" in self.monitor else self.monitor

    # --------------------------------------------------------------------- #
    def _train_step(self, batch: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        One optimisation step.  Returns (loss, log_dict).
        """
        batch = [x.to(self.cfg.device) for x in batch]
        loss, logs = self.model.loss(batch)

        loss.backward()
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        return loss.detach(), {k: float(v) for k, v in logs.items()}

    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def _validate_epoch(self) -> Dict[str, float]:
        """
        Cheap validation loop. Returns a dict of scalar metrics
        whose keys are prefixed with 'val/' for logging consistency.
        If no val_loader exists, returns an empty dict.
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        psnr_vals = []
        recon_losses = []

        for batch in self.val_loader:
            batch = [x.to(self.cfg.device) for x in batch]
            x = batch[0]
            x_hat = self.model(x)
            psnr_vals.append(psnr(x_hat, x).item())
            recon_losses.append(torch.nn.functional.mse_loss(x_hat, x).item())

        self.model.train()

        out = {
            "val/psnr": sum(psnr_vals) / len(psnr_vals),
            "val/loss": sum(recon_losses) / len(recon_losses),
        }
        return out

    # --------------------------------------------------------------------- #
    def _is_better(self, current: float) -> bool:
        if self.best_value is None:
            return True
        return (current < self.best_value) if self.mode == "min" else (current > self.best_value)

    # --------------------------------------------------------------------- #
    def _maybe_save_best(self, val_logs: Dict[str, float], epoch: int) -> None:
        """
        If current metric beats previous best, save ckpt and push to wandb.
        """
        if self._monitor_key not in val_logs:
            return  # nothing to compare

        current = val_logs[self._monitor_key]
        if self._is_better(current):
            self.best_value = current
            self.best_path = self.ckpt_dir / f"best_epoch{epoch}.pt"
            torch.save(self.model.state_dict(), self.best_path)

            # also store as wandb artifact so remote copy is preserved
            if wandb.run:
                wandb.save(str(self.best_path))
                wandb.run.summary[f"best_{self._monitor_key}"] = current

    # --------------------------------------------------------------------- #
    def fit(self) -> None:
        global_step = 0

        for epoch in range(self.cfg.trainer.max_epochs):
            pbar = tqdm(self.train_loader, dynamic_ncols=True, desc=f"epoch {epoch}")
            for batch in pbar:
                loss, logs = self._train_step(batch)

                # ── wandb logging ───────────────────────────────────────── #
                log_metrics(step=global_step, loss=loss.item(), **logs)
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                global_step += 1
            # end epoch train loop

            # ── validation & best-ckpt check ───────────────────────────── #
            val_logs = self._validate_epoch()
            if val_logs:
                log_metrics(step=global_step, **val_logs)  # all val/ metrics
                self._maybe_save_best(val_logs, epoch)

        # after all epochs, save final weights
        final_ckpt = self.ckpt_dir / "last.pt"
        torch.save(self.model.state_dict(), final_ckpt)
        if wandb.run:
            wandb.save(str(final_ckpt))
