from config import DATA_DIR, RESULTS_DIR, MODEL_PATH

import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from svetovid import (
    ConvAutoencoder,
    MarsAcceptableDataset,
    collect_image_paths,
    compute_batch_reconstruction_errors,
    save_json,
    set_seed,
)

# =========================================================
# MAIN SETTINGS
# =========================================================


IMAGE_SIZE = 256
BATCH_SIZE = 32
RANDOM_SEED = 42
NUM_WORKERS = 0

# =========================================================
# REPRODUCIBILITY
# =========================================================
set_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================================================
# DATA
# =========================================================
all_image_paths = collect_image_paths(DATA_DIR)
print(f"Collected images: {len(all_image_paths)}")

if len(all_image_paths) == 0:
    raise ValueError("No images found in DATA_DIR.")

dataset = MarsAcceptableDataset(
    image_paths=all_image_paths,
    image_size=IMAGE_SIZE
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

# =========================================================
# LOAD MODEL
# =========================================================
model = ConvAutoencoder().to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"Loaded model from: {MODEL_PATH}")

# =========================================================
# INFERENCE
# =========================================================
errors = []
records = []

total_start_time = time.time()
batch_times = []

with torch.no_grad():
    for inputs, paths in loader:
        inputs = inputs.to(device, non_blocking=True)
        batch_start = time.time()
        outputs = model(inputs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_end = time.time()
        batch_times.append(batch_end - batch_start)

        batch_errors = compute_batch_reconstruction_errors(inputs, outputs)

        for path, err in zip(paths, batch_errors.cpu().numpy().tolist()):
            errors.append(err)
            records.append({
                "file_path": path,
                "reconstruction_error": float(err)
            })

total_end_time = time.time()
total_inference_time = total_end_time - total_start_time
avg_inference_time = total_inference_time / len(records)
images_per_second = len(records) / total_inference_time
avg_batch_time = np.mean(batch_times)

# =========================================================
# RESULTS
# =========================================================
errors_np = np.array(errors)
mean_error = float(np.mean(errors_np))
std_error = float(np.std(errors_np))
min_error = float(np.min(errors_np))
max_error = float(np.max(errors_np))

print(f"Mean error: {mean_error:.6f}")
print(f"Std error:  {std_error:.6f}")
print(f"Min error:  {min_error:.6f}")
print(f"Max error:  {max_error:.6f}")

save_json(records, os.path.join(RESULTS_DIR, "inference_records.json"))
save_json(
    {
        "num_images": len(records),
        "mean_error": mean_error,
        "std_error": std_error,
        "min_error": min_error,
        "max_error": max_error,
        "total_inference_time_per_image_sec": avg_inference_time,
        "images_per_second": images_per_second,
        "avg_batch_time_sec": float(avg_batch_time)
    },
    os.path.join(RESULTS_DIR, "inference_summary.json")
)

print("Inference finished.")