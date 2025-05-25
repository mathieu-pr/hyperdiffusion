import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from utils.metrics import psnr
import torch.nn.functional as F
from utils.logger import log_metrics


class Evaluator:
    def __init__(self, model, datamodule, cfg, run_dir: Path):
        self.model = model.eval().to(cfg.device)
        self.loader = DataLoader(datamodule.train, batch_size=cfg.eval.batch_size)  # change datamodule.train -> datamodule.test
        self.cfg = cfg
        self.run_dir = run_dir

    @torch.no_grad()
    def run(self, split="test"):
        # psnrs = []
        mse_list = []
        for batch in self.loader:
            batch = [x.to(self.cfg.device) for x in batch]
            x_input = batch[0]
            x_output = self.model(x_input)
            mse = F.mse_loss(x_output, x_input, reduction='none').mean(dim=[1, 2, 3] if x_input.ndim == 4 else [1])
            mse_list.append(mse)
            # x̂ = self.model(batch[0])
            # psnrs.append(psnr(x̂, batch[0]))
        # mean_psnr = torch.stack(psnrs).mean().item()
        mean_mse = torch.cat(mse_list).mean().item()

        # log once to wandb (step=None writes to X-axis “undefined”)
        # log_metrics(psnr_test=mean_psnr)
        log_metrics(mse_test=mean_mse)

        out_path = self.run_dir / "eval.json"
        out_path.write_text(json.dumps({"mse": mean_mse}, indent=2))
        return {"mse": mean_mse}
