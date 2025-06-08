# src/engine/trainer.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import copy

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
    def __init__(self, model, splits, cfg, run_name: str, normalization_stats_path=None):
        print("-Version-: periodic backup of the best model, early stopping save")
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
        self.ckpt_folder = f"{timestamp}_{run_name}"                    # ADDED THIS!
        self.ckpt_dir = Path(
            cfg.trainer.ckpt_dir,
            cfg.trainer.model_name,
            self.ckpt_folder,
        )
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # ------------ Update default_eval.yaml file -----#


        self.monitor = cfg.trainer.monitor          # e.g. "val_recon_epoch"
        self.mode = cfg.trainer.mode.lower()        # "min" or "max"
        assert self.mode in {"min", "max"}
        self.best_value: float | None = None
        self.best_path: Path | None = None

        self.best_state_dict: dict[str, torch.Tensor] | None = None   # NEW
        self._last_backup_epoch: int | None = None                    # NEW
        self._monitor_key = (
            self.monitor.split("/", 1)[1] if "/" in self.monitor else self.monitor
        )

        # ---------------- Early-stopping ---------------- #
        self.patience = int(cfg.trainer.patience)   # NEW (early-stop)
        self._epochs_without_improve = 0            # NEW (early-stop)

        print(f"Patience for early stopping: {self.patience}, mode of comparison is: {self.mode}.\n")


        # ---------------- Normalization stats ------------- #
        if normalization_stats_path:
            stats = torch.load(normalization_stats_path)
            self.mean = stats["mean"].to(self.device)
            self.std = stats["std"].to(self.device)
        else:
            self.mean = None
            self.std = None

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
            # print(f"shape of batch: {[x.shape for x in batch]}")
            # print(f"len(batch) = {len(batch)}")
            batch = [x.to(self.cfg.device) for x in batch]
            # print(f"shape of batch after to(device): {[x.shape for x in batch]}")
            # print(f"len(batch) after to(device) = {len(batch)}")
            x = batch[0]
            ################################## print("\n\nx:", x, "\n\n") COMMENT THIS OUT
            # print(f"shape of x: {x.shape}")
            # print(f"len(x) = {len(x)}")
            x_hat = self.model(x)

            #unnormalize
            x_hat = x_hat * self.std + self.mean
            x = x * self.std + self.mean

            recon_losses.append(torch.nn.functional.mse_loss(x_hat, x).item())
            # print("val torch mse default:", torch.nn.functional.mse_loss(x_hat, x).item())
            # print("val torch mse with reduction='mean':", torch.nn.functional.mse_loss(x_hat, x, reduction='mean').item())
            # print("val torch mse with reduction='sum':", torch.nn.functional.mse_loss(x_hat, x, reduction='sum').item())
        self.model.train()
        print(f"recon_losses: {recon_losses}")
        print(f"len(recon_losses) = {len(recon_losses)}")
        print(f"Validation recon loss: {sum(recon_losses) / len(recon_losses):.4f}\n\n")

        return {
            "val_recon_epoch": sum(recon_losses) / len(recon_losses) if recon_losses else None,
        }

    # ------------------------------------------------------------------ #
    def _is_better(self, current: float) -> bool:
        if self.best_value is None:
            return True
        return (current < self.best_value) if self.mode == "min" else (current > self.best_value)

    # ------------------------------------------------------------------ #
    def _maybe_save_best(self, val_logs: Dict[str, float], epoch: int) -> None:
        """
        Save a checkpoint iff: (not used for now)
          • the monitored metric improved AND
          • epoch >= 10  (requested behaviour)
        """
        if self._monitor_key not in val_logs:
            print(f"\nWarning: {self._monitor_key} not found in validation logs.")
            print(f"Available keys: {list(val_logs.keys())}\n")
            return

        current = val_logs[self._monitor_key]
        improvement = self._is_better(current)
        best_value_str = f"{self.best_value:.4f}" if self.best_value is not None else "N/A"
        print(f".....Epoch {epoch}: {self._monitor_key} = {current:.4f} (best: {best_value_str})")
        # -------- bookkeeping for early stopping & in-RAM copy -------- #
        if improvement:
            print(f"Improvement detected: {self._monitor_key} improved from {best_value_str} to {current:.4f}")
            self.best_value = current
            self.best_state_dict = copy.deepcopy(
                {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
            )                                          # NEW
            self._epochs_without_improve = 0
        else:
            self._epochs_without_improve += 1
        # ---------- write the checkpoint when IMPROVED --------------- #
        if (not improvement) or epoch < 0:
            return
        
        # loss_str = f"{current:.4f}".replace(".", "_")
        # self.best_path = self.ckpt_dir / f"best_epoch{epoch}_vl{loss_str}.pt"
        # torch.save(self.best_state_dict, self.best_path)              # CHANGED            # CHANGED
    
        # print(f"New best {self._monitor_key} at epoch {epoch}: {current:.4f}")
        # print(f"Saved to {self.best_path}\n")

        # if wandb.run:
        #     artifact = wandb.Artifact("best_model", type="model")
        #     artifact.add_file(str(self.best_path))
        #     wandb.log_artifact(artifact)
        #     wandb.run.summary[f"best_{self._monitor_key}"] = current

    # ------------------------------------------------------------------ #
    def _should_stop_early(self) -> bool:            # NEW (early-stop)
        """Return True when patience is exceeded."""
        return self.patience > 0 and self._epochs_without_improve >= self.patience



    # ------------------------------------------------------------------ #
    def _periodic_backup(self, epoch: int) -> None:                    # NEW
        """Every 5 epochs write the in-RAM best snapshot, unless we already
        wrote one during this epoch."""
        if self.best_state_dict is None:
            return
        if (epoch+1) % 5 != 0:      # epochs start at 0 ⇒ 0,5,10,… OR use (epoch+1)%5==0
            return
        if self._last_backup_epoch == epoch:   # already saved because metric improved
            return
        loss_str = f"{self.best_value:.4f}".replace(".", "_")
        path = self.ckpt_dir / f"backup_epoch{epoch}_vl{loss_str}.pt"
        torch.save(self.best_state_dict, path)
        self._last_backup_epoch = epoch
        print(f"Periodic backup written to {path}")


    # ------------------------------------------------------------------ #
    def _save_best_on_early_stop(self) -> str:                        # NEW (UPDATED 6.4 >> return the best epoch name)
        """Write best weights to the final filename when early stopping kicks in."""
        if self.best_state_dict is None:
            print("Early stop, but no best_state_dict found – nothing saved.")
            return
        loss_str = f"{self.best_value:.4f}".replace(".", "_")
        best_ckpt_name = f"best_model_run_vl{loss_str}.pt"
        path = self.ckpt_dir / best_ckpt_name
        torch.save(self.best_state_dict, path)
        print(f"Early-stopping: best model (vl={self.best_value:.4f}) saved to {path}")
        return best_ckpt_name


    # ------------------------------------------------------------------ #
    def fit(self) -> str:
        global_step = 0
        best_epoch_name = None

        for epoch in range(self.cfg.trainer.max_epochs):
            train_loss_epoch_list = []
            pbar = tqdm(self.train_loader, desc=f"epoch {epoch}", dynamic_ncols=True)

            for batch in pbar:
                loss, logs = self._train_step(batch)
                # log_metrics(step=global_step, loss=loss.item(), **logs)
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                global_step += 1
                train_loss_epoch_list.append(loss.item())

            # ------------------ validation ------------------ #
            val_logs = self._validate_epoch()
            if val_logs:
                log_metrics(step=global_step, **val_logs, epoch=epoch + 1)
                self._maybe_save_best(val_logs, epoch)
            
            self._periodic_backup(epoch) 

            # ---------------- early-stopping ---------------- #
            if self._should_stop_early():  
                best_epoch_name = self._save_best_on_early_stop()         
                print(
                    f"Stopping early after epoch {epoch} – "
                    f"no improvement in {self.patience} epochs."
                )
                break

            # ----------- full-train reconstruction ---------- #
            self.model.eval()
            train_recon_losses = []
            print(f"\n\nFull train set: {len(self.train_loader)} batches")
            print(f"self.train_loader: {self.train_loader}")
            print(f"self.train_loader.dataset: {self.train_loader.dataset}")
            with torch.no_grad():
                for train_batch in self.train_loader:
                    # print(f"shape of train_batch: {[x.shape for x in train_batch]}")
                    # print(f"len(train_batch) = {len(train_batch)}")
                    train_batch = [x.to(self.cfg.device) for x in train_batch]
                    x = train_batch[0]
                    x_hat = self.model(x)

                    #unnormalize
                    x_hat = x_hat * self.std + self.mean
                    x = x * self.std + self.mean


                    # # print(f"len(x) = {len(x)}")
                    # print(f"len(x_hat) = {len(x_hat)}")
                    # print("torch mse default:", torch.nn.functional.mse_loss(x_hat, x).item())
                    # print("torch mse with reduction='mean':", torch.nn.functional.mse_loss(x_hat, x, reduction='mean').item())
                    # print("torch mse with reduction='sum':", torch.nn.functional.mse_loss(x_hat, x, reduction='sum').item())
                    train_recon_losses.append(torch.nn.functional.mse_loss(x_hat, x).item())
            self.model.train()
            # print(f"train_recon_losses: {train_recon_losses}")
            train_loss_epoch = sum(train_recon_losses) / len(train_recon_losses)
            # print(f"Full train recon loss: {train_loss_epoch:.4f}\n")
            # print(f"len(train_recon_losses) = {len(train_recon_losses)}")
            # print("\n\n")
            log_metrics(step=global_step, train_loss_epoch=train_loss_epoch, epoch=epoch + 1)


        # save final weights (the loop may have broken early)
        final_epoch = epoch   # last finished epoch
        final_ckpt = self.ckpt_dir / f"last_epoch{final_epoch}.pt"
        torch.save(self.model.state_dict(), final_ckpt)

        # if wandb.run:
        #     artifact = wandb.Artifact("final_model", type="model")
        #     artifact.add_file(str(final_ckpt))
        #     wandb.log_artifact(artifact)


        return self.ckpt_folder
