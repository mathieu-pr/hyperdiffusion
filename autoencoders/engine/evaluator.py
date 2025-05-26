import os
os.environ["HYDRA_FULL_ERROR"] = "1"

import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from utils.metrics import psnr
import torch.nn.functional as F
from utils.logger import log_metrics
from typing import Tuple, Optional, List
import trimesh

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from hyperdiffusion import HyperDiffusion


class Evaluator:
    def __init__(self, model, datamodule, cfg, run_dir: Path, hyperdiffusion_obj=None):
        self.model = model.eval().to(cfg.device)
        self.loader = DataLoader(datamodule.test, batch_size=cfg.eval.batch_size)  # change datamodule.train -> datamodule.test
        self.cfg = cfg
        self.run_dir = run_dir
        self.hyperdiffusion_obj = hyperdiffusion_obj

    def chamfer_distance(self, x: torch.Tensor, y: torch.Tensor):
        # Ensure x and y are [B, N, 3]
        #print(f"x.shape = {x.shape}")
        #print(f"y.shape = {y.shape}")
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if y.dim() == 2:
            y = y.unsqueeze(0)
        
        B, N, _ = x.shape
        _, M, _ = y.shape
        
        # Compute pairwise distance
        x_exp = x.unsqueeze(2)  # [B, N, 1, 3]
        y_exp = y.unsqueeze(1)  # [B, 1, M, 3]
        dist = torch.norm(x_exp - y_exp, dim=-1)  # [B, N, M]

        min_x_to_y = dist.min(dim=2)[0]  # [B, N]
        min_y_to_x = dist.min(dim=1)[0]  # [B, M]

        chamfer = (min_x_to_y.mean(dim=1) + min_y_to_x.mean(dim=1)) / 2  # [B]
        return chamfer.mean(), chamfer  # mean across batch, and per-batch scores

    @torch.no_grad()
    def run(self, split="test"):
        # psnrs = []
        mse_list = []
        chamfer_list = []
        
        for i, batch in enumerate(self.loader):  
            if i >= 2: # limit to 1 batch for debugging, remove condition for full evaluation
                break
            batch = [x.to(self.cfg.device) for x in batch]
            x_input = batch[0]
            x_output = self.model(x_input)
            mse = F.mse_loss(x_output, x_input, reduction='none').mean(dim=[1, 2, 3] if x_input.ndim == 4 else [1])
            mse_list.append(mse)
            # x̂ = self.model(batch[0])
            # psnrs.append(psnr(x̂, batch[0]))

            ## ----------- CHAMFER + RECONSTRUCTION ------------------ ##
            batch_chamfer = []

            if self.hyperdiffusion_obj is not None:
                # 1. Generate reconstructed meshes from AE output weights
                meshes_reconstructed, _ = self.hyperdiffusion_obj.generate_meshes(x_output, folder_name=None, res=64)

                # 2. Get ground truth meshes for this batch
                gt_meshes, _ = self.hyperdiffusion_obj.generate_meshes(x_input, folder_name=None, res=64)

                # 3. Compute chamfer distance batch-wise
                for rec_mesh, gt_mesh in zip(meshes_reconstructed, gt_meshes):
                    # Sample points from meshes (point clouds)
                    rec_points = torch.tensor(rec_mesh.sample(2048)).to(self.cfg.device)
                    gt_points = torch.tensor(gt_mesh.sample(2048)).to(self.cfg.device)

                    # Compute chamfer distance
                    chamfer_val, _ = self.chamfer_distance(rec_points.unsqueeze(0), gt_points.unsqueeze(0))
                    batch_chamfer.append(chamfer_val.item())

                # Store mean chamfer for this batch
                if batch_chamfer:
                    chamfer_list.append(torch.tensor(batch_chamfer).mean())
                    print(f"Batch {i} Chamfer Distance: {chamfer_list[-1].item()} - (batch size: {len(batch_chamfer)})")


            '''for i in range(x_output.shape[0]):
                pred_weights = x_output[i].unsqueeze(0)
                gt_weights = x_input[i].unsqueeze(0)

                # Generate meshes from predicted weights
                pred_meshes, _ = self.hyperdiffusion_obj.generate_meshes(pred_weights, None, res=700)  
                pred_mesh = pred_meshes[0] if pred_meshes else None

                # Generate meshes from ground-truth input (assuming same format)
                gt_meshes, _ = self.hyperdiffusion_obj.generate_meshes(gt_weights, None, res=700)
                gt_mesh = gt_meshes[0] if gt_meshes else None

                if pred_mesh is None or gt_mesh is None:
                    continue  # skip empty mesh cases

                # Sample point clouds from meshes for Chamfer Distance
                pred_pc = torch.tensor(pred_mesh.sample(2048)).to(self.cfg.device)
                gt_pc = torch.tensor(gt_mesh.sample(2048)).to(self.cfg.device)

                # Compute Chamfer Distance (using PyTorch3D)
                cd_loss, _ = self.chamfer_distance(pred_pc.unsqueeze(0), gt_pc.unsqueeze(0))
                batch_chamfer.append(cd_loss.item())

            
            
            if batch_chamfer:
                chamfer_list.append(torch.tensor(batch_chamfer).to(self.cfg.device))
            '''
            ## ----------- CHAMFER + RECONSTRUCTION ------------------ ##
        # mean_psnr = torch.stack(psnrs).mean().item()
        mean_mse = torch.cat(mse_list).mean().item()
        mean_chamfer = sum(chamfer_list) / len(chamfer_list) if chamfer_list else None
        # mean_chamfer = torch.cat(chamfer_list).mean().item() if chamfer_list else float('nan')

        # Convert mean_chamfer to float if it's a tensor
        if isinstance(mean_chamfer, torch.Tensor):
            mean_chamfer = mean_chamfer.item()

        print(f"mean_mse: {mean_mse}, mean_chamfer: {mean_chamfer}")

        # log once to wandb (step=None writes to X-axis “undefined”)
        # log_metrics(psnr_test=mean_psnr)
        log_metrics(mse_test=mean_mse, chamfer_test=mean_chamfer)

        out_path = self.run_dir / "eval.json"
        out_path.write_text(json.dumps({"mse": mean_mse, "chamfer": mean_chamfer}, indent=2))

        return {"mse": mean_mse, "chamfer": mean_chamfer}
    
    
