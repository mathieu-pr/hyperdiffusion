import itertools
import subprocess
import yaml
import csv
from pathlib import Path

# Hyperparameter grid
latent_dims = [1024] #[32, 64, 128]
hidden_dims_grid = [
    [4096, 512],
    #[1024, 256, 128],
    #[512, 256, 128, 64],
]
dropout_grid = [
    [0.1] * 2,
    #[0.1] * 3,
    #[0.2] * 4,
]
learning_rates = [1e-4] #[1e-3, 1e-4]
weight_decays = [1e-5] #[0.0, 1e-5]
lr_factors = [0.1]
lr_patiences = [5]

# Config file paths
config_dir = Path("autoencoders/configs")
ae_yaml_path = config_dir / "model/ae.yaml"
default_yaml_path = config_dir / "default.yaml"
results_csv = Path("grid_search_results.csv")

# Run counter and tracking best config
best_loss = float("inf")
best_config = {}

# Prepare CSV file
if not results_csv.exists():
    with results_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_name", "latent_dim", "hidden_dims", "dropout",
            "learning_rate", "weight_decay", "lr_factor", "lr_patience", "final_loss"
        ])

# Generate all combinations
grid = list(itertools.product(
    latent_dims, hidden_dims_grid, dropout_grid,
    learning_rates, weight_decays, lr_factors, lr_patiences
))

for i, (latent_dim, hidden_dims, dropout, lr, wd, factor, patience) in enumerate(grid):
    run_name = f"ld{latent_dim}_hd{'-'.join(map(str, hidden_dims))}_do{dropout[0]}_lr{lr}_wd{wd}"
    print(f"\n🚀 Running {i+1}/{len(grid)}: {run_name}")

    # Write ae.yaml
    ae_config = {
        "_target_": "autoencoders.models.ae.AutoEncoder",
        "input_dim": 36737,  # <-- Replace with your actual input dimension
        "latent_dim": latent_dim,
        "hidden_dims": hidden_dims,
        "dropout": dropout
    }
    with open(ae_yaml_path, "w") as f:
        yaml.dump(ae_config, f, sort_keys=False)

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

        # ⬇️ Replace this with how you track loss, e.g., from a file or wandb log
        final_loss = 0.1234  # ← Temporary placeholder
        print(f"✅ Run succeeded: final_loss = {final_loss}")

        # Update best config
        if final_loss < best_loss:
            best_loss = final_loss
            best_config = {
                "run_name": run_name,
                "latent_dim": latent_dim,
                "hidden_dims": hidden_dims,
                "dropout": dropout,
                "learning_rate": lr,
                "weight_decay": wd,
                "lr_scheduler_factor": factor,
                "lr_scheduler_patience": patience,
                "final_loss": final_loss,
            }

    except subprocess.CalledProcessError:
        print(f"❌ Run failed: {run_name}")
        final_loss = "FAILED"

    # Append to CSV
    with results_csv.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            run_name, latent_dim, str(hidden_dims), str(dropout),
            lr, wd, factor, patience, final_loss
        ])

# ✅ Print best config
print("\n🏆 Best config:")
print(yaml.dump(best_config, sort_keys=False))