import os
import shutil
import pandas as pd

import argparse
_p = argparse.ArgumentParser(description="Triage images into acceptable / rover_heavy / uncertain by classifier confidence")
_p.add_argument("--csv", required=True, help="CSV produced by predict_rover_classifier.py")
_p.add_argument("--output-dir", required=True)
_args = _p.parse_args()


# =========================
# USTAWIENIA
# =========================
CSV_PATH = _args.csv

OUTPUT_BASE = _args.output_dir

ACCEPTABLE_DIR = os.path.join(OUTPUT_BASE, "acceptable")
ROVER_DIR = os.path.join(OUTPUT_BASE, "rover_heavy")
UNCERTAIN_DIR = os.path.join(OUTPUT_BASE, "uncertain")

THRESHOLD_HIGH = 0.75
THRESHOLD_LOW = 0.25

# =========================
# TWORZENIE FOLDERÓW
# =========================
os.makedirs(ACCEPTABLE_DIR, exist_ok=True)
os.makedirs(ROVER_DIR, exist_ok=True)
os.makedirs(UNCERTAIN_DIR, exist_ok=True)

# =========================
# WCZYTANIE CSV
# =========================
df = pd.read_csv(CSV_PATH)

print(f"Total images: {len(df)}")

# =========================
# FUNKCJA KOPIOWANIA
# =========================
def copy_file(src, dst_folder):
    try:
        filename = os.path.basename(src)
        dst = os.path.join(dst_folder, filename)

        # unikamy nadpisywania
        counter = 1
        while os.path.exists(dst):
            name, ext = os.path.splitext(filename)
            dst = os.path.join(dst_folder, f"{name}_{counter}{ext}")
            counter += 1

        shutil.copy2(src, dst)

    except Exception as e:
        print(f"Error copying {src}: {e}")

# =========================
# SEGREGACJA
# =========================
count_acc = 0
count_rover = 0
count_uncertain = 0

for _, row in df.iterrows():
    path = row["file_path"]

    prob_acc = row["prob_acceptable"]
    prob_rover = row["prob_rover_heavy"]

    if prob_rover >= THRESHOLD_HIGH:
        copy_file(path, ROVER_DIR)
        count_rover += 1

    elif prob_rover <= THRESHOLD_LOW:
        copy_file(path, ACCEPTABLE_DIR)
        count_acc += 1

    else:
        copy_file(path, UNCERTAIN_DIR)
        count_uncertain += 1

print("\nDone.")
print(f"Acceptable: {count_acc}")
print(f"Rover heavy: {count_rover}")
print(f"Uncertain: {count_uncertain}")