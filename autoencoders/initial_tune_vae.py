# initial_tune_vae.py

import itertools
import subprocess
import yaml
import csv
from pathlib import Path

# Hyperparameter grid
latent_dims = [32, 128, 256]  # Example values
hidden_dims_grid = [
    [8192, 512],
    [8192, 2048, 512]
]
dropout_grid = [
    [0.3, 0.1],
    [0.3, 0.2, 0.1]
]
learning_rates = [1e-3]
weight_decays = [1e-5]
lr_factors = [0.1]
lr_patiences = [5]
recon_weights = [1.0]
kl_weights = [0.05, 0.1, 0.5]

# Config file paths
config_dir = Path("autoencoders/configs")
vae_yaml_path = config_dir / "model/vae.yaml"
default_yaml_path = config_dir / "default.yaml"
results_csv = Path("grid_search_results_vae.csv")

# Run counter and tracking best config
best_loss = float("inf")
best_config = {}

# Prepare CSV file
if not results_csv.exists():
    with results_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_name", "latent_dim", "hidden_dims", "dropout",
            "recon_weight", "kl_weight",
            "learning_rate", "weight_decay",
            "lr_factor", "lr_patience", "final_loss"
        ])

# Generate all combinations
grid = list(itertools.product(
    latent_dims, hidden_dims_grid, dropout_grid,
    recon_weights, kl_weights,
    learning_rates, weight_decays, lr_factors, lr_patiences
))

for i, (latent_dim, hidden_dims, dropout, rw, kw, lr, wd, factor, patience) in enumerate(grid):
    run_name = f"vae_ld{latent_dim}_hd{'-'.join(map(str, hidden_dims))}_do{dropout[0]}_rw{rw}_kw{kw}_lr{lr}_wd{wd}"
    print(f"\n🚀 Running {i+1}/{len(grid)}: {run_name}")

    # Write vae.yaml
    vae_config = {
        "_target_": "autoencoders.models.vae.VAE",
        "name": "vae",
        "latent_dim": latent_dim,
        "hidden_dims": hidden_dims,
        "dropout": dropout,
        "recon_weight": rw,
        "kl_weight": kw
    }
    with open(vae_yaml_path, "w") as f:
        yaml.dump(vae_config, f, sort_keys=False)

    # Write default.yaml
    with open(default_yaml_path, "r") as f:
        default_config = yaml.safe_load(f)
    default_config.update({
        "run_name": run_name,
        "learning_rate": lr,
        "weight_decay": wd,
        "lr_scheduler": {"factor": factor, "patience": patience},
    })
    with open(default_yaml_path, "w") as f:
        yaml.dump(default_config, f, sort_keys=False)

    # Run training
    try:
        subprocess.run(["python", "autoencoders/train.py"], check=True)

        # Replace with actual loss tracking
        with open("final_loss.txt", "r") as f:
            final_loss = float(f.read().strip())
        print(f"✅ Run succeeded: final_loss = {final_loss}")

        if final_loss < best_loss:
            best_loss = final_loss
            best_config = {
                "run_name": run_name,
                "latent_dim": latent_dim,
                "hidden_dims": hidden_dims,
                "dropout": dropout,
                "recon_weight": rw,
                "kl_weight": kw,
                "learning_rate": lr,
                "weight_decay": wd,
                "lr_scheduler_factor": factor,
                "lr_scheduler_patience": patience,
                "final_loss": final_loss,
            }

    except subprocess.CalledProcessError:
        print(f"❌ Run failed: {run_name}")
        final_loss = "FAILED"

    # Log result
    with results_csv.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            run_name, latent_dim, str(hidden_dims), str(dropout),
            rw, kw, lr, wd, factor, patience, final_loss
        ])

# ✅ Print best config
print("\n🏆 Best config:")
print(yaml.dump(best_config, sort_keys=False))