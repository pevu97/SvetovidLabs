import os
import shutil
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import random_split

import argparse
_p = argparse.ArgumentParser(description="Reproduce the 80/10/10 split (seed 42) and export the test set")
_p.add_argument("--data-dir", required=True)
_p.add_argument("--output-dir", required=True)
_args = _p.parse_args()


# =========================================================
# USTAWIENIA
# =========================================================
DATA_DIR = _args.data_dir
OUTPUT_TEST_DIR = _args.output_dir

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
RANDOM_SEED = 42

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

# =========================================================
# REPRODUKOWALNOŚĆ
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_SEED)

# =========================================================
# POMOCNICZE
# =========================================================
def is_image_file(filename: str) -> bool:
    return filename.lower().endswith(IMAGE_EXTENSIONS)

def collect_image_paths(root_dir: str):
    paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if is_image_file(file):
                paths.append(os.path.join(root, file))
    paths.sort()
    return paths

# =========================================================
# ODTWORZENIE SPLITU
# =========================================================
all_image_paths = collect_image_paths(DATA_DIR)
print(f"Collected images: {len(all_image_paths)}")

if len(all_image_paths) == 0:
    raise ValueError("No images found in DATA_DIR.")

total_len = len(all_image_paths)
train_len = int(total_len * TRAIN_RATIO)
val_len = int(total_len * VAL_RATIO)
test_len = total_len - train_len - val_len

print(f"Expected split sizes:")
print(f"Train: {train_len}")
print(f"Val:   {val_len}")
print(f"Test:  {test_len}")

generator = torch.Generator().manual_seed(RANDOM_SEED)

indices_dataset = list(range(total_len))
train_subset, val_subset, test_subset = random_split(
    indices_dataset,
    [train_len, val_len, test_len],
    generator=generator
)

test_indices = test_subset.indices
test_paths = [all_image_paths[i] for i in test_indices]

print(f"Recovered test paths: {len(test_paths)}")

# =========================================================
# KOPIOWANIE DO JEDNEGO FOLDERU
# =========================================================
os.makedirs(OUTPUT_TEST_DIR, exist_ok=True)

copied = 0
name_conflicts = 0

for src_path in test_paths:
    filename = os.path.basename(src_path)
    dst_path = os.path.join(OUTPUT_TEST_DIR, filename)

    # zabezpieczenie, gdyby dwa pliki miały taką samą nazwę
    if os.path.exists(dst_path):
        name_conflicts += 1
        stem = Path(filename).stem
        suffix = Path(filename).suffix
        parent_name = Path(src_path).parent.name
        dst_path = os.path.join(OUTPUT_TEST_DIR, f"{stem}__{parent_name}{suffix}")

    shutil.copy2(src_path, dst_path)
    copied += 1

print(f"Copied files: {copied}")
print(f"Filename conflicts resolved: {name_conflicts}")
print(f"Done. Test set copied to: {OUTPUT_TEST_DIR}")