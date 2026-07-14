import os
import csv
import shutil
import hashlib
from collections import defaultdict
from PIL import Image
import imagehash

import argparse
_p = argparse.ArgumentParser(description="Remove exact (SHA-256) and near (phash) duplicates")
_p.add_argument("--input-dir", required=True, help="Dataset root to deduplicate in place")
_args = _p.parse_args()



# ====== USTAWIENIA ======
BASE_FOLDER = _args.input_dir

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

# ile kolejnych obrazów porównywać dla near-duplicates
WINDOW_SIZE = 5

# próg podobieństwa phash
MAX_DISTANCE = 2

EXACT_DUP_FOLDER_NAME = "_removed_exact_duplicates"
NEAR_DUP_FOLDER_NAME = "_removed_near_duplicates"

EXACT_REPORT_NAME = "exact_duplicates_report.csv"
NEAR_REPORT_NAME = "near_duplicates_report.csv"
# ========================


def is_image_file(filename):
    return filename.lower().endswith(IMAGE_EXTENSIONS)


def should_skip_path(path):
    skip_folders = {
        EXACT_DUP_FOLDER_NAME.lower(),
        NEAR_DUP_FOLDER_NAME.lower()
    }

    parts = [p.lower() for p in path.split(os.sep)]
    return any(part in skip_folders for part in parts)


def get_all_image_paths(base_folder):
    image_paths = []

    for root, _, files in os.walk(base_folder):
        if should_skip_path(root):
            continue

        for file in files:
            if is_image_file(file):
                image_paths.append(os.path.join(root, file))

    # stabilna kolejność
    image_paths.sort()
    return image_paths


def file_hash(filepath, chunk_size=8192):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def make_relative_subpath(base_folder, file_path):
    return os.path.relpath(file_path, base_folder)


def safe_move_file(src_path, dst_root, base_folder):
    rel_path = make_relative_subpath(base_folder, src_path)
    dst_path = os.path.join(dst_root, rel_path)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    if os.path.exists(dst_path):
        base, ext = os.path.splitext(dst_path)
        counter = 1
        while os.path.exists(f"{base}__dup{counter}{ext}"):
            counter += 1
        dst_path = f"{base}__dup{counter}{ext}"

    shutil.move(src_path, dst_path)
    return dst_path


def save_csv(rows, csv_path, header):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def remove_exact_duplicates(base_folder, image_paths):
    exact_removed_root = os.path.join(base_folder, EXACT_DUP_FOLDER_NAME)
    os.makedirs(exact_removed_root, exist_ok=True)

    hash_map = defaultdict(list)

    print("Liczenie hashy exact duplicates...")
    for i, path in enumerate(image_paths, start=1):
        try:
            h = file_hash(path)
            hash_map[h].append(path)
        except Exception as e:
            print(f"Błąd hash dla {path}: {e}")

        if i % 1000 == 0:
            print(f"  Przeanalizowano {i}/{len(image_paths)} plików")

    report_rows = []
    removed_count = 0
    duplicate_groups = 0
    kept_paths = []

    print("Przenoszenie exact duplicates...")
    for h, paths in hash_map.items():
        paths = sorted(paths)

        if len(paths) == 1:
            kept_paths.append(paths[0])
            continue

        duplicate_groups += 1
        keep_file = paths[0]
        kept_paths.append(keep_file)

        report_rows.append([duplicate_groups, h, keep_file, "kept", ""])

        for dup_path in paths[1:]:
            try:
                moved_to = safe_move_file(dup_path, exact_removed_root, base_folder)
                report_rows.append([duplicate_groups, h, dup_path, "moved", moved_to])
                removed_count += 1
            except Exception as e:
                report_rows.append([duplicate_groups, h, dup_path, f"error: {e}", ""])

    exact_report_path = os.path.join(base_folder, EXACT_REPORT_NAME)
    save_csv(
        report_rows,
        exact_report_path,
        header=["group_id", "hash", "file_path", "action", "moved_to"]
    )

    print("\n=== EXACT DUPLICATES ===")
    print(f"Liczba grup duplikatów: {duplicate_groups}")
    print(f"Liczba przeniesionych plików: {removed_count}")
    print(f"Raport: {exact_report_path}")
    print("========================\n")

    kept_paths.sort()
    return kept_paths


def compute_phashes(image_paths):
    hashes = {}

    print("Liczenie perceptual hash (phash)...")
    for i, path in enumerate(image_paths, start=1):
        try:
            with Image.open(path) as img:
                phash = imagehash.phash(img)
                hashes[path] = phash
        except Exception as e:
            print(f"Błąd phash dla {path}: {e}")

        if i % 1000 == 0:
            print(f"  Przeanalizowano {i}/{len(image_paths)} plików")

    return hashes


def remove_near_duplicates(base_folder, image_paths, phashes, window_size=5, max_distance=2):
    near_removed_root = os.path.join(base_folder, NEAR_DUP_FOLDER_NAME)
    os.makedirs(near_removed_root, exist_ok=True)

    to_remove = set()
    report_rows = []
    group_id = 0

    print("Wykrywanie near-duplicates (sliding window)...")

    n = len(image_paths)

    for i in range(n):
        current_path = image_paths[i]

        if current_path in to_remove:
            continue

        current_hash = phashes.get(current_path)
        if current_hash is None:
            continue

        current_group = [current_path]

        for j in range(i + 1, min(i + 1 + window_size, n)):
            candidate_path = image_paths[j]

            if candidate_path in to_remove:
                continue

            candidate_hash = phashes.get(candidate_path)
            if candidate_hash is None:
                continue

            distance = current_hash - candidate_hash

            if distance <= max_distance:
                current_group.append(candidate_path)

        if len(current_group) > 1:
            group_id += 1
            keep_file = current_group[0]
            report_rows.append([group_id, keep_file, "kept", "", 0])

            for dup_path in current_group[1:]:
                if dup_path in to_remove:
                    continue

                distance = current_hash - phashes[dup_path]
                to_remove.add(dup_path)
                report_rows.append([group_id, dup_path, "to_move", "", distance])

        if (i + 1) % 1000 == 0:
            print(f"  Sprawdzono {i + 1}/{n} plików")

    moved_count = 0

    print("Przenoszenie near-duplicates...")
    final_report_rows = []

    for row in report_rows:
        group_id_row, file_path, action, moved_to, distance = row

        if action == "kept":
            final_report_rows.append([group_id_row, file_path, "kept", "", distance])
            continue

        try:
            moved_path = safe_move_file(file_path, near_removed_root, base_folder)
            final_report_rows.append([group_id_row, file_path, "moved", moved_path, distance])
            moved_count += 1
        except Exception as e:
            final_report_rows.append([group_id_row, file_path, f"error: {e}", "", distance])

    near_report_path = os.path.join(base_folder, NEAR_REPORT_NAME)
    save_csv(
        final_report_rows,
        near_report_path,
        header=["group_id", "file_path", "action", "moved_to", "phash_distance"]
    )

    print("\n=== NEAR DUPLICATES ===")
    print(f"Liczba grup near-duplicates: {group_id}")
    print(f"Liczba przeniesionych plików: {moved_count}")
    print(f"Raport: {near_report_path}")
    print("=======================\n")


def main():
    print(f"Folder bazowy: {BASE_FOLDER}")
    image_paths = get_all_image_paths(BASE_FOLDER)
    print(f"Znaleziono obrazów: {len(image_paths)}")

    # Krok 1: exact duplicates
    image_paths_after_exact = remove_exact_duplicates(BASE_FOLDER, image_paths)

    # Krok 2: near duplicates w oknie sekwencyjnym
    phashes = compute_phashes(image_paths_after_exact)
    remove_near_duplicates(
        base_folder=BASE_FOLDER,
        image_paths=image_paths_after_exact,
        phashes=phashes,
        window_size=WINDOW_SIZE,
        max_distance=MAX_DISTANCE
    )

    print("Gotowe.")


if __name__ == "__main__":
    main()