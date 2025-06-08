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
    def __init__(self, model, datamodule, cfg, run_dir: Path, split = "test", hyperdiffusion_obj=None, normalization_stats_path=None):
        self.model = model.eval().to(cfg.device)
        self.split = split
        if split == "test" :
            self.loader = DataLoader(datamodule.test, batch_size=cfg.eval.batch_size)  # change datamodule.train -> datamodule.test (TRAIN FOR SANITY)
        elif split == "train":
            self.loader = DataLoader(datamodule.train, batch_size=cfg.eval.batch_size)
        self.cfg = cfg
        self.device = cfg.device
        self.run_dir = run_dir
        self.hyperdiffusion_obj = hyperdiffusion_obj

        if normalization_stats_path:
            stats = torch.load(normalization_stats_path)
            self.mean = stats["mean"].to(self.device)
            self.std = stats["std"].to(self.device)
        else :
            self.mean = None
            self.std = None

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
        return chamfer, chamfer.mean()  # mean across batch, and per-batch scores

    @torch.no_grad()
    def run(self):
        # psnrs = []
        mse_list = []
        chamfer_list = []
        if self.split == "test" :
            vis_dir = os.path.join(self.run_dir, "autoencoders/mesh_visualization")
        else :
            vis_dir = os.path.join(self.run_dir, "autoencoders/mesh_visualization/train")
        os.makedirs(vis_dir, exist_ok=True)
        
        for i, batch in enumerate(self.loader):  
            if i >= self.cfg.eval.max_batches_eval: 
                break
            batch = [x.to(self.cfg.device) for x in batch]
            x_input = batch[0]
            x_output = self.model(x_input)

            # UNNORMALIZATION :
            x_input = x_input * self.std + self.mean
            x_output = x_output * self.std + self.mean
            
            # MSE LOSS :
            mse = F.mse_loss(x_output, x_input, reduction='none').mean(dim=[1, 2, 3] if x_input.ndim == 4 else [1])
            mse_list.append(mse)
            # x̂ = self.model(batch[0])
            # psnrs.append(psnr(x̂, batch[0]))

            ## ----------- CHAMFER + RECONSTRUCTION ------------------ ##
            batch_chamfer = []

            if self.hyperdiffusion_obj is not None:
                # 1. Generate reconstructed meshes from AE output weights
                meshes_reconstructed, _ = self.hyperdiffusion_obj.generate_meshes(x_output, folder_name=None, res=256)

                # 2. Get ground truth meshes for this batch
                gt_meshes, _ = self.hyperdiffusion_obj.generate_meshes(x_input, folder_name=None, res=256)

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

                    # ----- Visualization -----
                    for j, (rec_mesh, gt_mesh) in enumerate(zip(meshes_reconstructed, gt_meshes)):
                        # Validate reconstructed mesh
                        try:
                            if rec_mesh is None or len(rec_mesh.vertices) == 0 or len(rec_mesh.faces) == 0:
                                print(f"Skipping empty reconstructed mesh for batch {i}, item {j}")
                                continue

                            if gt_mesh is None or len(gt_mesh.vertices) == 0 or len(gt_mesh.faces) == 0:
                                print(f"Skipping empty GT mesh for batch {i}, item {j}")
                                continue

                            rec_mesh_path = os.path.join(vis_dir, f"batch{i}_item{j}_recon.ply")
                            gt_mesh_path = os.path.join(vis_dir, f"batch{i}_item{j}_gt.ply")

                            rec_mesh.export(rec_mesh_path)
                            gt_mesh.export(gt_mesh_path)

                            # Optional visualization
                            scene = trimesh.Scene()
                            rec_mesh.visual.face_colors = [255, 0, 0, 100]  # Red, semi-transparent
                            gt_mesh.visual.face_colors = [0, 255, 0, 100]   # Green, semi-transparent
                            scene.add_geometry(gt_mesh)
                            scene.add_geometry(rec_mesh)
                            scene.show()
                        except Exception as e:
                            print(f"Could not render mesh: {e}")


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
    
    
