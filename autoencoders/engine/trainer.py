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


import warnings

warnings.filterwarnings(
    "ignore",
    message=".*Metric `FrechetInceptionDistance` will save all extracted features in buffer.*",
    category=UserWarning
)


class Trainer:
    def __init__(self, model, splits, cfg, run_name: str, normalization_stats_path=None):
        self.cfg = cfg
        self.device = torch.device(cfg.device)          
        self.model = model.to(self.device)
        self.run_name = run_name

        
        # ----------------- DataLoaders ------------------ #
        pin = self.device.type == "cuda"
        self.train_loader = DataLoader(
            splits.train,
            batch_size=cfg.trainer.batch_size,
            shuffle=True,
            num_workers=cfg.dataset.num_workers,
            pin_memory=pin,
            persistent_workers=True
        )
        self.val_loader = (
            DataLoader(
                splits.val,
                batch_size=cfg.trainer.batch_size,
                shuffle=False,
                num_workers=cfg.dataset.num_workers,
                pin_memory=pin,
                persistent_workers=True
            )
            if getattr(splits, "val", None) is not None
            else None
        )

        # ---------------- Optimiser --------------------- #
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.wd,
        )

        # ---------------- LR Scheduler ------------------- #
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt,
            mode="min",
            factor=cfg.scheduler.factor if hasattr(cfg, "scheduler") and hasattr(cfg.scheduler, "factor") else 0.5,
            patience=cfg.scheduler.patience if hasattr(cfg, "scheduler") and hasattr(cfg.scheduler, "patience") else 5,
            verbose=True
        )

        # ------------- Checkpoint bookkeeping ---------- #
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.ckpt_folder = f"{timestamp}_{run_name}"                   
        self.ckpt_dir = Path(
            cfg.trainer.ckpt_dir,
            cfg.model.name,
            self.ckpt_folder,
        )
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # ------------ Performance monitoring -----#
        self.monitor = cfg.trainer.monitor          # e.g. "val_recon_epoch"
        self.mode = cfg.trainer.mode.lower()        # "min" or "max"
        assert self.mode in {"min", "max"}
        self.best_value: float | None = None
        self.best_path: Path | None = None
        self.best_state_dict: dict[str, torch.Tensor] | None = None   
        self._last_backup_epoch: int | None = None                    
        self._monitor_key = (
            self.monitor.split("/", 1)[1] if "/" in self.monitor else self.monitor
        )


        # ---------------- Early-stopping ---------------- #
        self.patience = int(cfg.trainer.patience)   
        self._epochs_without_improve = 0           


        # ---------------- Normalization stats ------------- #
        if normalization_stats_path:
            stats = torch.load(normalization_stats_path)
            self.mean = stats["mean"].to(self.device)
            self.std = stats["std"].to(self.device)
        else:
            self.mean = None
            self.std = None
        
        # ------------------ Logging config info --------------------- #
        print("=" * 60)
        print("Trainer Initialization Summary")
        print("=" * 60)
        print(f"Run name:              {self.run_name}")
        print(f"Checkpoint directory:  {self.ckpt_dir}")
        print(f"Device:                {cfg.device}")
        print(f"Model parameters:      {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Early stopping:")
        print(f"  Patience:            {self.patience}")
        print(f"  Mode:                {self.mode}")
        print(f"Monitor metric:        {self.monitor}")
        print(f"Normalization stats:")
        print(f"  Mean:                {self.mean}")
        print(f"  Std:                 {self.std}")
        print(f"Training batch size:   {cfg.trainer.batch_size}")
        print(f"Number of workers:     {cfg.dataset.num_workers}")
        if hasattr(cfg, "scheduler"):
            print("LR Scheduler parameters:")
            factor = getattr(cfg.scheduler, "factor", None)
            patience = getattr(cfg.scheduler, "patience", None)
            print(f"  factor: {factor}")
            print(f"  patience: {patience}")
        else:
            print("No LR Scheduler configured.")
        print(f"Initial learning rate: {cfg.optimizer.lr}")
        print("=" * 60 + "\n")

        if self.val_loader is None:
            print("Warning: No validation set provided! Validation will be skipped.")

        #------------------ Logging dataset info --------------------- #   #for debug only
        log_dataset_info = True
        if log_dataset_info:
            print("=" * 60)
            print("Dataset Statistics")
            print("-" * 60)
            
            # Compute statistics for training set
            self._compute_dataset_statistics(self.train_loader, "Training")
            
            # Compute statistics for validation set if available
            if self.val_loader is not None:
                self._compute_dataset_statistics(self.val_loader, "Validation")
            else:
                print("  Validation set   | Not provided")
            
            print("=" * 60)


    # ------------------------------------------------------------------ #
    def _compute_dataset_statistics(self, loader: DataLoader, split_name: str) -> None:
        """
        Compute and print detailed statistics for a dataset.
        
        Args:
            loader: DataLoader for the dataset
            split_name: Name of the split ('Training' or 'Validation') for logging
        """
        dataset = loader.dataset
        dataset_len = len(dataset)
        inputs = []
        
        # Collect all inputs
        batch_size = 1024 if split_name == "Validation" else 128
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            inputs.append(x)
        inputs = torch.cat(inputs, dim=0)
        
        # Compute basic statistics
        mean = inputs.mean().item()
        var = inputs.var(unbiased=False).item()
        
        # Compute extremes
        flat = inputs.flatten()
        top10_max = torch.topk(flat, 10).values.cpu().numpy()
        top10_min = torch.topk(-flat, 10).values.cpu().numpy()
        top10_min = -top10_min
        
        # Find outliers
        #mask = (inputs > 1000).any(dim=tuple(range(1, inputs.dim())))
        mask = (inputs > 1000).view(inputs.size(0), -1).any(dim=1)
        indices = torch.nonzero(mask, as_tuple=False).squeeze().cpu().numpy()
        num_outliers = mask.sum().item()
        
        # Print statistics in a structured format
        print(f"  {split_name:15s} | Size: {dataset_len:6d} | Mean: {mean:10.6f} | Var: {var:10.6f}")
        print(f"    Top 10 max values: {[round(v, 2) for v in top10_max]}")
        print(f"    Top 10 min values: {[round(v, 2) for v in top10_min]}")
        print(f"    Samples with any parameter > 1000: {num_outliers}")
        print(f"    Indices of such samples: {indices}")
        print("-" * 60)


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
    def _compute_reconstruction_losses(self, loader: DataLoader, split_name: str, epoch: int | None = None) -> Dict[str, float]:
        """
        Compute reconstruction losses for a given loader.
        
        Args:
            loader: DataLoader to compute losses for
            split_name: Name of the split ('train' or 'val') for logging
            epoch: Current epoch number for logging
        
        Returns:
            Dictionary containing the average reconstruction loss
        """
        self.model.eval()
        recon_losses = []

        
        for batch in loader:
            batch = [x.to(self.cfg.device) for x in batch]
            x = batch[0]
            x_hat = self.model(x)

            # unnormalize
            x_hat = x_hat * self.std + self.mean
            x = x * self.std + self.mean

            recon_losses.append(torch.nn.functional.mse_loss(x_hat, x).item())
        
        # Compute statistics
        avg_loss = sum(recon_losses) / len(recon_losses)
        max_loss = max(recon_losses)
        min_loss = min(recon_losses)
        top_10_losses = sorted(recon_losses, reverse=True)[:10]
        
        # Print diagnostics in a clear, hierarchical format
        epoch_str = f" (epoch {epoch})" if epoch is not None else ""
        print("\n" + "=" * 70)
        print(f"[{split_name.upper()} SET RECONSTRUCTION LOSSES]{epoch_str}")
        print("-" * 70)
        print(f"  Number of batches:      {len(recon_losses)}")
        print(f"  Sum of losses:          {sum(recon_losses):.6f}")
        print(f"  Mean loss:              {avg_loss:.6f}")
        print(f"  Max loss:               {max_loss:.6f}")
        print(f"  Min loss:               {min_loss:.6f}")
        print(f"  Top 10 largest losses:  {[round(l, 3) for l in top_10_losses]}")
        print("-" * 70)
        return {
            f"{split_name}_recon_epoch": avg_loss
        }
    

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def compute_loss(self, loader: DataLoader, split_name: str = "val") -> Dict[str, float]:
        """
        Computes the average of all losses returned by the model's loss function over the entire dataloader.

        Args:
            loader (DataLoader): DataLoader for which to compute the loss.
            split_name (str): Used for naming keys in the returned dictionary (e.g., 'train' or 'val').

        Returns:
            Dict[str, float]: Dictionary of averaged loss components, prefixed by the split_name.
        """
        self.model.eval()
        loss_logs_accum: Dict[str, float] = {}
        count = 0

        for batch in loader:
            batch = [x.to(self.device) for x in batch]
            _, logs = self.model.loss(batch)

            for key, val in logs.items():
                loss_logs_accum[key] = loss_logs_accum.get(key, 0.0) + val
            count += 1

        averaged_logs = {f"{split_name}_{k}": v / count for k, v in loss_logs_accum.items()}
        return averaged_logs



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
          • epoch >= 10  #(to avoid saving too early, e.g. at epoch 0)
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
            )                                         
            self._epochs_without_improve = 0
        else:
            self._epochs_without_improve += 1
        
      

    # ------------------------------------------------------------------ #
    def _should_stop_early(self) -> bool:            #
        """Return True when patience is exceeded."""
        return self.patience > 0 and self._epochs_without_improve >= self.patience



    # ------------------------------------------------------------------ #
    def _periodic_backup(self, epoch: int) -> None:                  
        """Every 5 epochs write the in-RAM best snapshot, unless we already
        wrote one during this epoch."""
        if self.best_state_dict is None:
            return
        if (epoch+1) % 5 != 0:      
            return
        if self._last_backup_epoch == epoch:   # already saved because metric improved
            return
        loss_str = f"{self.best_value:.4f}".replace(".", "_")
        path = self.ckpt_dir / f"backup_epoch{epoch}_vl{loss_str}.pt"
        self.save_checkpoint_with_parameters(
            epoch=epoch,
            name="backup",
            val_recon=loss_str
        )
        self._last_backup_epoch = epoch


    # ------------------------------------------------------------------ #
    def _save_best_on_early_stop(self) -> str:                        
        """Write best weights to the final filename when early stopping kicks in."""
        if self.best_state_dict is None:
            print("Early stop, but no best_state_dict found – nothing saved.")
            return
        loss_str = f"{self.best_value:.4f}".replace(".", "_")
        best_ckpt_name = self.save_checkpoint_with_parameters(
            epoch="Earlystop",
            name="best_model_run",
            val_recon=loss_str
        )
        print(f"Early-stopping: best model (vl={self.best_value:.4f}) saved to {best_ckpt_name}")
        return best_ckpt_name
    

    # ------------------------------------------------------------------ #
    def save_checkpoint_with_parameters(self, epoch, name: str, val_recon):
        """Save a checkpoint with model parameters, validation reconstruction loss, and model config."""
        val_recon_str = f"{float(val_recon):.4f}"
        ckpt_path = self.ckpt_dir / f"{name}_epoch{epoch}_valrecon{val_recon_str}.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.opt.state_dict(),
            "epoch": epoch,
            "val_recon": val_recon,
            "model_cfg": self.cfg.model
        }, ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")
        return f"{name}_epoch{epoch}_valrecon{val_recon}.pt"


    # ------------------------------------------------------------------ #
    def fit(self) -> str:
        global_step = 0
        best_epoch_name = None

        for epoch in range(self.cfg.trainer.max_epochs):
            train_loss_epoch_list = []
            pbar = tqdm(self.train_loader, desc=f"epoch {epoch}", dynamic_ncols=True)

            self.model.train()
            # Accumulate logs for averaging
            epoch_logs = {}
            num_batches = 0

            for batch in pbar:
                loss, logs = self._train_step(batch)
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                global_step += 1
                train_loss_epoch_list.append(loss.item())
                num_batches += 1

                # Accumulate metrics
                for k, v in logs.items():
                    if k not in epoch_logs:
                        epoch_logs[k] = 0.0
                    epoch_logs[k] += v

            # Compute mean of metrics over the epoch
            mean_epoch_logs = {k: v / num_batches for k, v in epoch_logs.items()}
            log_metrics(step=global_step, **mean_epoch_logs, epoch=epoch + 1)

            # -------------- Log validation loss for all loss components ---------------- #
            val_losses = self.compute_loss(self.val_loader, "val")
            log_metrics(step=global_step, **val_losses, epoch=epoch + 1)

            # -------------- Log training and validation loss ---------------- #  
            train_logs = self._compute_reconstruction_losses(self.train_loader, "train", epoch)
            log_metrics(step=global_step, **train_logs, epoch=epoch + 1)

            val_logs = self._compute_reconstruction_losses(self.val_loader, "val", epoch) if self.val_loader else {}
            log_metrics(step=global_step, **val_logs, epoch=epoch + 1)

            # -------------- Log current learning rate ---------------- #
            current_lr = self.opt.param_groups[0]["lr"]
            print(f"Current learning rate: {current_lr:.6f}")
            log_metrics(step=global_step, lr=current_lr, epoch=epoch + 1)

            # ----------------- Scheduler step ----------------- #
            if self.val_loader and self.scheduler is not None:
                monitor_value = val_logs.get(self._monitor_key)
                if monitor_value is not None:
                    self.scheduler.step(monitor_value)
            else:
                print("No validation set provided or scheduler not configured. Skipping scheduler step.")

            # ----------------- Save best model if improved ----------------- #
            if self.val_loader:
                self._maybe_save_best(val_logs, epoch)
        
            # ----------------- Periodic backup ----------------- #
            self._periodic_backup(epoch) 

            # ---------------- early-stopping ---------------- #
            if self._should_stop_early():  
                best_epoch_name = self._save_best_on_early_stop()         
                print(
                    f"Stopping early after epoch {epoch} – "
                    f"no improvement in {self.patience} epochs."
                )
                break


        # ------------------ Save last epoch model ----------------- #
        final_epoch = epoch   # last finished epoch
        # Use the best_value if available, otherwise use 0.0 as a placeholder
        val_recon_value = self.best_value if self.best_value is not None else 0.0
        final_ckpt = self.save_checkpoint_with_parameters(
            epoch=final_epoch,
            name="last_epoch",
            val_recon=val_logs["val_recon_epoch"]
        )
        print(f"Last epoch model saved to {final_ckpt}")

        # ------------------ Save best model if not already saved ----------------- #   
        if best_epoch_name is None and self.best_state_dict is not None:
            best_epoch_name = self._save_best_on_early_stop()
        if best_epoch_name is not None:
            print(f"Best model already saved as {best_epoch_name} in {self.ckpt_dir}")

        return self.ckpt_folder

    