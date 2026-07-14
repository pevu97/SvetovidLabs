import os
import json
import csv

import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import MobileNet_V3_Small_Weights
from PIL import Image, ImageFile

import argparse
_p = argparse.ArgumentParser(description="Batch inference of the rover classifier over a dataset")
_p.add_argument("--image-dir", required=True)
_p.add_argument("--model", required=True)
_p.add_argument("--class-map", required=True)
_p.add_argument("--output-csv", required=True)
_args = _p.parse_args()


# =========================
# USTAWIENIA
# =========================
IMAGE_DIR = _args.image_dir
MODEL_PATH = _args.model
CLASS_MAP_PATH = _args.class_map
OUTPUT_CSV = _args.output_csv

IMAGE_SIZE = 224
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
SKIP_FOLDERS = {"acceptable", "rover_heavy", "_removed_exact_duplicates", "_removed_near_duplicates"}

ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# WCZYTANIE MAPOWANIA KLAS
# =========================
with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
    class_to_idx = json.load(f)

idx_to_class = {v: k for k, v in class_to_idx.items()}
print("class_to_idx:", class_to_idx)
print("idx_to_class:", idx_to_class)

# =========================
# MODEL
# =========================
weights = MobileNet_V3_Small_Weights.DEFAULT
model = models.mobilenet_v3_small(weights=None)  # przy wczytywaniu własnych wag nie trzeba pretrained
in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

print(f"Using device: {DEVICE}")

# =========================
# ZBIERANIE ŚCIEŻEK
# =========================
def is_image_file(filename):
    return filename.lower().endswith(IMAGE_EXTENSIONS)

def should_skip_path(path):
    parts = {p.lower() for p in path.split(os.sep)}
    return any(folder.lower() in parts for folder in SKIP_FOLDERS)

image_paths = []
for root, _, files in os.walk(IMAGE_DIR):
    if should_skip_path(root):
        continue

    for file in files:
        if is_image_file(file):
            image_paths.append(os.path.join(root, file))

image_paths.sort()
print(f"Znaleziono obrazów do inferencji: {len(image_paths)}")

# =========================
# INFERENCJA
# =========================
rows = []

softmax = nn.Softmax(dim=1)

for i, path in enumerate(image_paths, start=1):
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            x = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(x)
            probs = softmax(logits)[0].cpu().numpy()

        pred_idx = int(probs.argmax())
        pred_class = idx_to_class[pred_idx]

        prob_acceptable = float(probs[class_to_idx["acceptable"]])
        prob_rover_heavy = float(probs[class_to_idx["rover_heavy"]])

        rows.append([
            path,
            pred_class,
            prob_acceptable,
            prob_rover_heavy
        ])

    except Exception as e:
        rows.append([
            path,
            f"ERROR: {e}",
            "",
            ""
        ])

    if i % 500 == 0:
        print(f"Przetworzono {i}/{len(image_paths)} obrazów")

# =========================
# ZAPIS CSV
# =========================
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["file_path", "predicted_class", "prob_acceptable", "prob_rover_heavy"])
    writer.writerows(rows)

print(f"\nZapisano wyniki do: {OUTPUT_CSV}")
print("Gotowe.")