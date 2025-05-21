import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from utils.metrics import psnr
from utils.logger import log_metrics


class Evaluator:
    def __init__(self, model, datamodule, cfg, run_dir: Path):
        self.model = model.eval().to(cfg.device)
        self.loader = DataLoader(datamodule.test, batch_size=cfg.eval.batch_size)
        self.cfg = cfg
        self.run_dir = run_dir

    @torch.no_grad()
    def run(self, split="test"):
        psnrs = []
        for batch in self.loader:
            batch = [x.to(self.cfg.device) for x in batch]
            x̂ = self.model(batch[0])
            psnrs.append(psnr(x̂, batch[0]))
        mean_psnr = torch.stack(psnrs).mean().item()

        # log once to wandb (step=None writes to X-axis “undefined”)
        log_metrics(psnr_test=mean_psnr)

        out_path = self.run_dir / "eval.json"
        out_path.write_text(json.dumps({"psnr": mean_psnr}, indent=2))
        return {"psnr": mean_psnr}
