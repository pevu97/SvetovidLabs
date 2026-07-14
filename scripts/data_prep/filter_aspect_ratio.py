import os
import csv
import shutil
from PIL import Image

import argparse
_p = argparse.ArgumentParser(description="Move panoramic images (aspect ratio >= threshold) out of the dataset")
_p.add_argument("--input-dir", required=True)
_p.add_argument("--output-dir", required=True)
_args = _p.parse_args()



# ===== USTAWIENIA =====
INPUT_FOLDER = _args.input_dir
OUTPUT_FOLDER = _args.output_dir

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

# Próg proporcji width / height
ASPECT_RATIO_THRESHOLD = 2.0

# Foldery, które pomijamy przy przeszukiwaniu
SKIP_FOLDERS = {
    "_removed_exact_duplicates",
    "_removed_near_duplicates"
}

REPORT_CSV = "aspect_ratio_filter_report.csv"
# ======================


def is_image_file(filename):
    return filename.lower().endswith(IMAGE_EXTENSIONS)


def should_skip_path(path):
    parts = [p.lower() for p in path.split(os.sep)]
    return any(skip.lower() in parts for skip in SKIP_FOLDERS)


def safe_copy_file(src_path, dst_root, base_folder):
    rel_path = os.path.relpath(src_path, base_folder)
    dst_path = os.path.join(dst_root, rel_path)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    if os.path.exists(dst_path):
        base, ext = os.path.splitext(dst_path)
        counter = 1
        while os.path.exists(f"{base}__copy{counter}{ext}"):
            counter += 1
        dst_path = f"{base}__copy{counter}{ext}"

    shutil.move(src_path, dst_path)
    return dst_path


def save_csv(rows, csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file_path",
            "width",
            "height",
            "aspect_ratio",
            "decision",
            "copied_to"
        ])
        writer.writerows(rows)


def process_dataset():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    report_rows = []
    selected = 0
    skipped = 0
    errors = 0

    for root, _, files in os.walk(INPUT_FOLDER):
        if should_skip_path(root):
            continue

        for file in files:
            if not is_image_file(file):
                continue

            image_path = os.path.join(root, file)

            try:
                with Image.open(image_path) as img:
                    width, height = img.size

                if height == 0:
                    report_rows.append([image_path, width, height, "", "error_height_zero", ""])
                    errors += 1
                    continue

                aspect_ratio = width / height

                if aspect_ratio >= ASPECT_RATIO_THRESHOLD:
                    copied_to = safe_copy_file(image_path, OUTPUT_FOLDER, INPUT_FOLDER)
                    report_rows.append([
                        image_path,
                        width,
                        height,
                        round(aspect_ratio, 4),
                        "selected",
                        copied_to
                    ])
                    selected += 1
                else:
                    report_rows.append([
                        image_path,
                        width,
                        height,
                        round(aspect_ratio, 4),
                        "skipped",
                        ""
                    ])
                    skipped += 1

            except Exception as e:
                report_rows.append([image_path, "", "", "", f"error: {e}", ""])
                errors += 1

    csv_path = os.path.join(OUTPUT_FOLDER, REPORT_CSV)
    save_csv(report_rows, csv_path)

    print("\n==========================")
    print(f"Próg aspect ratio: {ASPECT_RATIO_THRESHOLD}")
    print(f"Wybrane zdjęcia: {selected}")
    print(f"Odrzucone zdjęcia: {skipped}")
    print(f"Błędy: {errors}")
    print(f"Raport CSV: {csv_path}")
    print("==========================")


if __name__ == "__main__":
    process_dataset()