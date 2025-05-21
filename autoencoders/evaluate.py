import hydra
from omegaconf import DictConfig
from engine.evaluator import Evaluator
from utils.logger import init_wandb


@hydra.main(config_path="../configs", config_name="train/default")
def main(cfg: DictConfig):
    run = init_wandb(cfg, run_name=f"eval_{cfg.ckpt_path.split('/')[-2]}")
    # 1. rebuild model & datamodule
    from models.registry import build_model
    from data.make_datamodule import make_datamodule

    model = build_model(cfg)
    model.load_state_dict(torch.load(cfg.ckpt_path, map_location="cpu"))

    datamodule = make_datamodule(cfg)
    evaluator = Evaluator(model, datamodule, cfg, run_dir=Path(cfg.ckpt_path).parent)
    evaluator.run(split=cfg.eval.split)
    run.finish()


if __name__ == "__main__":
    main()
