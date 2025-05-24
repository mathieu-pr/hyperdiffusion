# src/engine/trainer.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from utils.logger import log_metrics

from datetime import datetime


class Trainer:
    """
    Framework-free trainer.

    • expects obj.train (Dataset) and optional obj.val (Dataset)
    • builds its own DataLoader
    • logs via utils.logger.log_metrics (→ wandb)
    • saves best checkpoint according to cfg.trainer.monitor
    """

    # ------------------------------------------------------------------ #
    def __init__(self, model, splits, cfg, run_name: str):
        self.cfg = cfg
        print(f"Device: {cfg.device}")
        self.device = torch.device(cfg.device)          # ← NO fallback
        self.model = model.to(self.device)
        self.run_name = run_name

        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model has {num_params:,} parameters.\n")

        # ----------------- DataLoaders ------------------ #
        pin = self.device.type == "cuda"
        self.train_loader = DataLoader(
            splits.train,
            batch_size=cfg.trainer.batch_size,
            shuffle=True,
            num_workers=cfg.dataset.num_workers,
            pin_memory=pin,
        )
        self.val_loader = (
            DataLoader(
                splits.val,
                batch_size=cfg.trainer.batch_size,
                shuffle=False,
                num_workers=cfg.dataset.num_workers,
                pin_memory=pin,
            )
            if getattr(splits, "val", None) is not None
            else None
        )

        # ---------------- Optimiser --------------------- #
        self.opt = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.wd,
        )

        # ------------- Checkpoint bookkeeping ---------- #
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.ckpt_dir = Path(cfg.trainer.ckpt_dir, f"{cfg.trainer.model_name}", f"{timestamp}_{run_name}")
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = cfg.trainer.monitor
        self.mode = cfg.trainer.mode.lower()
        assert self.mode in {"min", "max"}
        self.best_value: float | None = None
        self.best_path: Path | None = None
        self._monitor_key = (
            self.monitor.split("/", 1)[1] if "/" in self.monitor else self.monitor
        )

    # ------------------------------------------------------------------ #
    def _train_step(self, batch: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch = [x.to(self.cfg.device) for x in batch]
        loss, logs = self.model.loss(batch)

        loss.backward()
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        return loss.detach(), {k: float(v) for k, v in logs.items()}

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _validate_epoch(self) -> Dict[str, float]:
        if self.val_loader is None:
            print("!No validation set provided!")
            return {}

        self.model.eval()
        recon_losses = []
        for batch in self.val_loader:
            batch = [x.to(self.cfg.device) for x in batch]
            x = batch[0]
            x_hat = self.model(x)
            # recon_losses.append(torch.nn.functional.mse_loss(x_hat, x).item())
            recon_losses.append(torch.nn.functional.mse_loss(x_hat, x).item())
        self.model.train()

        return {
            "val_recon_epoch": sum(recon_losses) / len(recon_losses),
        }

    # ------------------------------------------------------------------ #
    def _is_better(self, current: float) -> bool:
        if self.best_value is None:
            return True
        return (current < self.best_value) if self.mode == "min" else (current > self.best_value)

    # ------------------------------------------------------------------ #
    def _maybe_save_best(self, val_logs: Dict[str, float], epoch: int) -> None:
        if self._monitor_key not in val_logs:
            return
        current = val_logs[self._monitor_key]
        if self._is_better(current):
            self.best_value = current
            self.best_path = self.ckpt_dir / f"best_epoch{epoch}.pt"
            torch.save(self.model.state_dict(), self.best_path)
            if wandb.run:
                artifact = wandb.Artifact("best_model", type="model")
                artifact.add_file(str(self.best_path))
                wandb.log_artifact(artifact)
                wandb.run.summary[f"best_{self._monitor_key}"] = current

    # ------------------------------------------------------------------ #
    def fit(self) -> None:
        print("New with loss on full train set at the end of each epoch")
        global_step = 0
        for epoch in range(self.cfg.trainer.max_epochs):
            
            train_loss_epoch_list = []
            pbar = tqdm(self.train_loader, desc=f"epoch {epoch}", dynamic_ncols=True)
            for batch in pbar:
                loss, logs = self._train_step(batch)
                log_metrics(step=global_step, loss=loss.item(), **logs)
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                global_step += 1
                train_loss_epoch_list.append(loss.item())

            val_logs = self._validate_epoch()
            if val_logs:
                log_metrics(step=global_step, **val_logs, epoch=epoch+1)
                self._maybe_save_best(val_logs, epoch)

            #log the train_loss_epoch   
            # train_loss_epoch = sum(train_loss_epoch_list) / len(train_loss_epoch_list)
            # log_metrics(step=global_step, train_loss_epoch=train_loss_epoch, epoch=epoch+1)

            # Evaluate the model on the full training set
            self.model.eval()
            train_recon_losses = []
            with torch.no_grad():
                for train_batch in self.train_loader:
                    train_batch = [x.to(self.cfg.device) for x in train_batch]
                    x = train_batch[0]
                    x_hat = self.model(x)
                    train_recon_losses.append(torch.nn.functional.mse_loss(x_hat, x).item())
            self.model.train()
            train_loss_epoch = sum(train_recon_losses) / len(train_recon_losses)
            log_metrics(step=global_step, train_loss_epoch=train_loss_epoch, epoch=epoch+1)

        # save final weights
        final_epoch = self.cfg.trainer.max_epochs - 1
        final_ckpt = self.ckpt_dir / f"last_epoch{final_epoch}.pt"
        torch.save(self.model.state_dict(), final_ckpt)

        # if wandb.run:    #use wandb to log the final model (disabled to save space)
        #     artifact = wandb.Artifact("final_model", type="model")
        #     artifact.add_file(str(final_ckpt))
        #     wandb.log_artifact(artifact)

