from config import INFERENCE_RECORDS_PATH, INFERENCE_SUMMARY_PATH, SIMULATION_RESULTS_DIR

import pandas as pd
import os
import json
import math
import shutil
from pathlib import Path



with open(INFERENCE_SUMMARY_PATH, "r", encoding="utf-8") as f:
    inference_summary = json.load(f)


# =========================================================
# UTILITY FUNCTIONS
# =========================================================
def print_inference_summary(summary: dict):
    print("=== INFERENCE PERFORMANCE ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("=============================")



def load_inference_records(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nie znaleziono pliku: {path}")

    df = pd.read_json(path)

    required_columns = {"file_path", "reconstruction_error"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Brakuje wymaganych kolumn w inference_records.json: {missing_columns}"
        )

    return df


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def copy_selected_images(df_selected: pd.DataFrame, destination_dir: str) -> dict:
    ensure_dir(destination_dir)

    copied_count = 0
    missing_count = 0
    renamed_count = 0

    for _, row in df_selected.iterrows():
        src_path = row["file_path"]

        if not os.path.exists(src_path):
            missing_count += 1
            continue

        filename = os.path.basename(src_path)
        dst_path = os.path.join(destination_dir, filename)

        # zabezpieczenie na wypadek duplikatów nazw
        if os.path.exists(dst_path):
            renamed_count += 1
            stem = Path(filename).stem
            suffix = Path(filename).suffix
            parent_name = Path(src_path).parent.name
            dst_path = os.path.join(destination_dir, f"{stem}__{parent_name}{suffix}")

        shutil.copy2(src_path, dst_path)
        copied_count += 1

    return {
        "copied_files": copied_count,
        "missing_source_files": missing_count,
        "renamed_due_to_name_conflict": renamed_count
    }


def simulate_transmission(df: pd.DataFrame, percent: int, output_dir: str) -> dict:
    if percent <= 0 or percent > 100:
        raise ValueError("percent musi być w zakresie 1-100")

    df_sorted = df.sort_values("reconstruction_error", ascending=False).reset_index(drop=True)

    total_images = len(df_sorted)
    selected_count = math.ceil(total_images * (percent / 100.0))

    df_selected = df_sorted.head(selected_count).copy()
    df_selected["rank"] = range(1, len(df_selected) + 1)
    df_selected["transmission_percent"] = percent
    df_selected["selected_for_transmission"] = True

    csv_path = os.path.join(output_dir, f"scenario_{percent}pct.csv")
    df_selected.to_csv(csv_path, index=False)

    selected_images_dir = os.path.join(output_dir, f"selected_{percent}pct")
    copy_stats = copy_selected_images(df_selected, selected_images_dir)

    metrics = {
        "scenario_name": f"{percent}pct",
        "total_images": int(total_images),
        "selected_images": int(selected_count),
        "discarded_images": int(total_images - selected_count),
        "transmission_percent": float(percent),
        "reduction_percent": float(100.0 - percent),
        "total_inference_time_per_image": float(inference_summary.get("total_inference_time_per_image_sec", 0)),
        "inference_per_second": float(inference_summary.get("images_per_second", 0)),
        "avg_batch_time_sec": float(inference_summary.get("avg_batch_time_sec", 0)),
        "selected_score_mean": float(df_selected["reconstruction_error"].mean()),
        "selected_score_min": float(df_selected["reconstruction_error"].min()),
        "selected_score_max": float(df_selected["reconstruction_error"].max()),
        "output_csv": csv_path,
        "selected_images_dir": selected_images_dir,
        "copy_stats": copy_stats
    }

    return metrics


# =========================================================
# MAIN
# =========================================================
def main():
    ensure_dir(SIMULATION_RESULTS_DIR)

    print_inference_summary(inference_summary)

    print("Loading inference results...")
    df_inference = load_inference_records(INFERENCE_RECORDS_PATH)
    print(f"Number of inference records: {len(df_inference)}")

    scenarios = [10, 5, 1]
    all_metrics = []

    for percent in scenarios:
        print(f"\nSimulating transmission for scenario: {percent}%")

        metrics = simulate_transmission(
            df=df_inference,
            percent=percent,
            output_dir=SIMULATION_RESULTS_DIR
        )
        all_metrics.append(metrics)

        print(
            f"Scenariusz {percent}% | "
            f"selected={metrics['selected_images']} | "
            f"discarded={metrics['discarded_images']} | "
            f"reduction={metrics['reduction_percent']:.2f}%"
        )
        print(
            f"Skopiowano: {metrics['copy_stats']['copied_files']} | "
            f"brakujących źródeł: {metrics['copy_stats']['missing_source_files']} | "
            f"zmienionych nazw: {metrics['copy_stats']['renamed_due_to_name_conflict']}"
        )

    summary_path = os.path.join(SIMULATION_RESULTS_DIR, "transmission_summary.json")
    
    final_summary = {
        "inference_metrics": inference_summary,
        "transmission_scenarios": all_metrics
    }

    save_json(final_summary, summary_path)

    print(f"\nSummary saved to: {summary_path}")
    print("Simulation finished.")


if __name__ == "__main__":
    main()