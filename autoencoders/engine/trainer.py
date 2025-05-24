# src/engine/trainer.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from utils.logger import log_metrics


class Trainer:
    """
    Framework-free trainer.

    • expects obj.train (Dataset) and optional obj.val (Dataset)
    • builds its own DataLoader
    • logs via utils.logger.log_metrics (→ wandb)
    • saves best checkpoint according to cfg.trainer.monitor
    • NEW: early stopping controlled by cfg.trainer.patience
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
        self.ckpt_dir = Path(
            cfg.trainer.ckpt_dir,
            cfg.trainer.model_name,
            f"{timestamp}_{run_name}",
        )
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = cfg.trainer.monitor          # e.g. "val_recon_epoch"
        self.mode = cfg.trainer.mode.lower()        # "min" or "max"
        assert self.mode in {"min", "max"}
        self.best_value: float | None = None
        self.best_path: Path | None = None
        self._monitor_key = (
            self.monitor.split("/", 1)[1] if "/" in self.monitor else self.monitor
        )

        # ---------------- Early-stopping ---------------- #
        self.patience = int(cfg.trainer.patience)   # NEW (early-stop)
        self._epochs_without_improve = 0            # NEW (early-stop)

        print(f"Patience for early stopping: {self.patience}\n")

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
        """
        Save a checkpoint iff:
          • the monitored metric improved AND
          • epoch >= 20  (requested behaviour)
        """
        if self._monitor_key not in val_logs:
            return

        current = val_logs[self._monitor_key]
        improvement = self._is_better(current)

        # -------- bookkeeping for early stopping -------- #
        if improvement:
            self.best_value = current
            self._epochs_without_improve = 0        # NEW (early-stop)
        else:
            self._epochs_without_improve += 1       # NEW (early-stop)

        if not improvement or epoch < 20:           # NEW (best-ckpt condition)
            return

        # ---------- write the checkpoint --------------- #
        # filename: best_epoch{epoch}_vl{val_loss:.4f}.pt
        loss_str = f"{current:.4f}".replace(".", "_")     # avoid dots in filename
        self.best_path = self.ckpt_dir / f"best_epoch{epoch}_vl{loss_str}.pt"
        torch.save(self.model.state_dict(), self.best_path)

        if wandb.run:
            artifact = wandb.Artifact("best_model", type="model")
            artifact.add_file(str(self.best_path))
            wandb.log_artifact(artifact)
            wandb.run.summary[f"best_{self._monitor_key}"] = current

    # ------------------------------------------------------------------ #
    def _should_stop_early(self) -> bool:            # NEW (early-stop)
        """Return True when patience is exceeded."""
        return self.patience > 0 and self._epochs_without_improve >= self.patience

    # ------------------------------------------------------------------ #
    def fit(self) -> None:
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

            # ------------------ validation ------------------ #
            val_logs = self._validate_epoch()
            if val_logs:
                log_metrics(step=global_step, **val_logs, epoch=epoch + 1)
                self._maybe_save_best(val_logs, epoch)

            # ---------------- early-stopping ---------------- #
            if self._should_stop_early():            # NEW (early-stop)
                print(
                    f"Stopping early after epoch {epoch} – "
                    f"no improvement in {self.patience} epochs."
                )
                break

            # ----------- full-train reconstruction ---------- #
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
            log_metrics(step=global_step, train_loss_epoch=train_loss_epoch, epoch=epoch + 1)

        # save final weights (the loop may have broken early)
        final_epoch = epoch   # last finished epoch
        final_ckpt = self.ckpt_dir / f"last_epoch{final_epoch}.pt"
        torch.save(self.model.state_dict(), final_ckpt)

        # if wandb.run:
        #     artifact = wandb.Artifact("final_model", type="model")
        #     artifact.add_file(str(final_ckpt))
        #     wandb.log_artifact(artifact)
