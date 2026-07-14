"""Train the ConvAutoencoder on curated nominal NAVCAM images.

Faithful script version of the original training pipeline
(train/val/test = 80/10/10, L1 loss, Adam + weight decay,
ReduceLROnPlateau, light augmentation, AMP on CUDA, early stopping),
extended with MLflow experiment tracking.

After training, the script:
  * computes the anomaly threshold on the validation set (mean + k*std),
  * evaluates reconstruction errors on the held-out test set,
  * saves per-epoch reconstruction previews and the loss curve.

Experiment tracking is stored in a local SQLite database (mlflow.db)
by default. Inspect the runs with the UI pointed at the same database:

Usage:
    python train.py --data-dir data --epochs 40
    mlflow ui --backend-store-uri sqlite:///mlflow.db   # inspect runs
"""

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from config import DATA_DIR, MODEL_PATH
from svetovid import (
    ConvAutoencoder,
    MarsAcceptableDataset,
    collect_image_paths,
    compute_batch_reconstruction_errors,
    save_json,
    set_seed,
)

# =========================================================
# ARGUMENTS (defaults = original training configuration)
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train Svetovid ConvAutoencoder")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--output-dir", type=str, default="ae_results")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=7,
                        help="Early stopping patience (epochs without val improvement)")
    parser.add_argument("--threshold-std-factor", type=float, default=3.0,
                        help="Anomaly threshold = val_mean + k * val_std")
    parser.add_argument("--num-recon-vis", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default=str(MODEL_PATH),
                        help="Path for the best checkpoint")
    parser.add_argument("--experiment", type=str, default="svetovid-autoencoder")
    parser.add_argument("--tracking-uri", type=str, default="sqlite:///mlflow.db",
                        help="MLflow tracking store (default: local SQLite database)")
    return parser.parse_args()


# =========================================================
# VISUALIZATION
# =========================================================
def plot_history(train_losses, val_losses, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("L1 loss")
    plt.title("Training history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_reconstructions(inputs, outputs, epoch, save_dir, max_images=8):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    inputs = inputs.detach().cpu()
    outputs = outputs.detach().cpu()

    n = min(max_images, inputs.size(0))
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    if n == 1:
        axes = np.array([axes])

    for i in range(n):
        axes[i, 0].imshow(inputs[i].squeeze(0).numpy(), cmap="gray")
        axes[i, 0].set_title("Input")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(outputs[i].squeeze(0).numpy(), cmap="gray")
        axes[i, 1].set_title("Reconstruction")
        axes[i, 1].axis("off")

    plt.tight_layout()
    path = save_dir / f"epoch_{epoch:03d}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def collect_errors(model, loader, device):
    errors, records = [], []
    with torch.no_grad():
        for inputs, paths in loader:
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            batch_errors = compute_batch_reconstruction_errors(inputs, outputs)
            for path, err in zip(paths, batch_errors.cpu().numpy().tolist()):
                errors.append(err)
                records.append({"file_path": path, "reconstruction_error": float(err)})
    return errors, records


# =========================================================
# MAIN
# =========================================================
def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data: 80/10/10 split, augmentation on train only ----
    all_image_paths = collect_image_paths(args.data_dir)
    print(f"Collected images: {len(all_image_paths)}")
    if len(all_image_paths) < 3:
        raise ValueError(f"Not enough images in {args.data_dir}.")

    total_len = len(all_image_paths)
    train_len = int(total_len * args.train_ratio)
    val_len = int(total_len * args.val_ratio)
    test_len = total_len - train_len - val_len

    generator = torch.Generator().manual_seed(args.seed)
    train_subset, val_subset, test_subset = random_split(
        list(range(total_len)), [train_len, val_len, test_len], generator=generator
    )

    train_paths = [all_image_paths[i] for i in train_subset.indices]
    val_paths = [all_image_paths[i] for i in val_subset.indices]
    test_paths = [all_image_paths[i] for i in test_subset.indices]

    train_dataset = MarsAcceptableDataset(train_paths, args.image_size, augment=True)
    val_dataset = MarsAcceptableDataset(val_paths, args.image_size, augment=False)
    test_dataset = MarsAcceptableDataset(test_paths, args.image_size, augment=False)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # ---- Model / loss / optimizer / scheduler ----
    model = ConvAutoencoder().to(device)
    # L1 usually gives sharper reconstructions than pure MSE
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

    # ---- MLflow ----
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    train_losses, val_losses = [], []
    checkpoint_path = Path(args.checkpoint)

    with mlflow.start_run():
        mlflow.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "image_size": args.image_size,
            "early_stopping_patience": args.patience,
            "scheduler": "ReduceLROnPlateau(factor=0.5, patience=3)",
            "augmentation": "hflip(p=0.5) + rotation(5deg)",
            "amp": scaler is not None,
            "optimizer": "Adam",
            "loss": "L1",
            "threshold_std_factor": args.threshold_std_factor,
            "seed": args.seed,
            "train_images": len(train_dataset),
            "val_images": len(val_dataset),
            "test_images": len(test_dataset),
            "device": str(device),
        })

        start_time = time.time()

        for epoch in range(1, args.epochs + 1):
            # ---- train ----
            model.train()
            running_train_loss = 0.0
            for inputs, _paths in train_loader:
                inputs = inputs.to(device, non_blocking=True)
                optimizer.zero_grad()

                if scaler is not None:
                    with torch.amp.autocast(device_type="cuda"):
                        outputs = model(inputs)
                        loss = criterion(outputs, inputs)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, inputs)
                    loss.backward()
                    optimizer.step()

                running_train_loss += loss.item() * inputs.size(0)

            epoch_train_loss = running_train_loss / len(train_dataset)
            train_losses.append(epoch_train_loss)

            # ---- validate ----
            model.eval()
            running_val_loss = 0.0
            last_val_inputs = last_val_outputs = None
            with torch.no_grad():
                for inputs, _paths in val_loader:
                    inputs = inputs.to(device, non_blocking=True)
                    outputs = model(inputs)
                    loss = criterion(outputs, inputs)
                    running_val_loss += loss.item() * inputs.size(0)
                    last_val_inputs, last_val_outputs = inputs, outputs

            epoch_val_loss = running_val_loss / len(val_dataset)
            val_losses.append(epoch_val_loss)

            scheduler.step(epoch_val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            mlflow.log_metrics(
                {"train_loss": epoch_train_loss,
                 "val_loss": epoch_val_loss,
                 "lr": current_lr},
                step=epoch,
            )
            print(f"Epoch [{epoch}/{args.epochs}] | "
                  f"train_loss: {epoch_train_loss:.6f} | "
                  f"val_loss: {epoch_val_loss:.6f} | "
                  f"lr: {current_lr:.7f}")

            if last_val_inputs is not None:
                save_reconstructions(
                    last_val_inputs, last_val_outputs, epoch,
                    output_dir / "reconstructions", args.num_recon_vis,
                )

            # ---- best checkpoint / early stopping ----
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                        "image_size": args.image_size,
                    },
                    checkpoint_path,
                )
                print(f"Saved best model at epoch {epoch}")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

        training_time = time.time() - start_time

        # ---- history ----
        curve_path = output_dir / "loss_history.png"
        plot_history(train_losses, val_losses, curve_path)

        # ---- reload best model ----
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print(f"Loaded best model from epoch: {checkpoint['epoch']} "
              f"(val {checkpoint['val_loss']:.6f})")

        # ---- anomaly threshold on validation errors ----
        val_errors, _ = collect_errors(model, val_loader, device)
        val_mean = float(np.mean(val_errors))
        val_std = float(np.std(val_errors))
        anomaly_threshold = val_mean + args.threshold_std_factor * val_std
        print(f"VAL error mean: {val_mean:.6f} | std: {val_std:.6f} | "
              f"anomaly threshold: {anomaly_threshold:.6f}")

        # ---- held-out test evaluation ----
        test_errors, test_records = collect_errors(model, test_loader, device)
        for record in test_records:
            record["is_anomaly_by_threshold"] = bool(
                record["reconstruction_error"] > anomaly_threshold
            )
        test_mean = float(np.mean(test_errors))
        test_std = float(np.std(test_errors))
        print(f"TEST error mean: {test_mean:.6f} | std: {test_std:.6f}")

        thresholds_path = output_dir / "thresholds.json"
        save_json(
            {
                "val_error_mean": val_mean,
                "val_error_std": val_std,
                "test_error_mean": test_mean,
                "test_error_std": test_std,
                "anomaly_threshold": anomaly_threshold,
                "threshold_formula": f"mean + {args.threshold_std_factor} * std",
            },
            thresholds_path,
        )
        save_json(test_records[:500], output_dir / "test_records_sample.json")

        # ---- final MLflow logging ----
        mlflow.log_metrics({
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "val_error_mean": val_mean,
            "val_error_std": val_std,
            "anomaly_threshold": anomaly_threshold,
            "test_error_mean": test_mean,
            "test_error_std": test_std,
            "training_time_sec": training_time,
        })
        mlflow.log_artifact(str(curve_path))
        mlflow.log_artifact(str(thresholds_path))
        mlflow.log_artifact(str(checkpoint_path))

    print("Training pipeline finished.")


if __name__ == "__main__":
    main()
