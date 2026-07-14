# %%
from config import INFERENCE_RECORDS_PATH, INFERENCE_SUMMARY_PATH, TRANSMISSION_SUMMARY_PATH, REPORT_DIR, DATA_DIR

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
from PIL import Image

if not os.path.exists(REPORT_DIR):
    os.makedirs(REPORT_DIR)

df_inference_records = pd.read_json(INFERENCE_RECORDS_PATH)
df_inference_summary = pd.read_json(INFERENCE_SUMMARY_PATH, typ='series')
df_transmission_summary = pd.read_json(TRANSMISSION_SUMMARY_PATH, typ='series')

errors = df_inference_records['reconstruction_error']

treshold_10 = errors.quantile(0.9)
treshold_5 = errors.quantile(0.95)
treshold_1 = errors.quantile(0.99)

tresholds = {
    '10%': treshold_10,
    '5%': treshold_5,
    '1%': treshold_1
}

images = {
    '10 pct': df_transmission_summary['transmission_scenarios'][0]['selected_images_dir'],
    '5 pct': df_transmission_summary['transmission_scenarios'][1]['selected_images_dir'],
    '1 pct': df_transmission_summary['transmission_scenarios'][2]['selected_images_dir']
}

rejected_images = []

random_rejected_images = []
n_selected = int(np.ceil(len(df_inference_records) * 0.1))

sorted_images = df_inference_records.sort_values(by="reconstruction_error", ascending=False)
accepted_images = sorted_images.head(n_selected)
rejected_images = sorted_images.iloc[n_selected:]

random_rejected_images = rejected_images['file_path'].sample(
    n=min(6, len(rejected_images)),
    random_state=42
).tolist()

lowest_rejected_images = rejected_images.sort_values(
    by="reconstruction_error",
    ascending=True
)['file_path'].head(6).tolist()

print(n_selected)
print(len(accepted_images))
print(len(rejected_images))

def generate_report():
    print(f'\n=== ANALYSIS REPORT ===\n')
    print(f'Total images: {len(errors)}')
    print(f'Mean error: {errors.mean():.4f}')
    print(f'Std error: {errors.std():.4f}')

    print(f'\nTresholds:\n')
    for key, value in tresholds.items():
        print(f'{key}: {value:.4f}')
    

def hist_error():
    plt.figure(figsize=(12, 6))
    sns.histplot(errors, bins=50, kde=True)

    plt.axvline(treshold_10, color='orange', linestyle='--', label='10% threshold')
    plt.axvline(treshold_5, color='red', linestyle='--', label='5% threshold')
    plt.axvline(treshold_1, color='purple', linestyle='--', label='1% threshold')

    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.title("Reconstruction Error Distribution with KDE")
    plt.legend()
    plt.savefig(os.path.join(REPORT_DIR, 'error_histogram.png'))
    plt.show()

def transmission_comparision():
    columns_to_rename = {
        'scenario_name': 'Scenario',
        'total_images': 'Total Images',
        'selected_images': 'Selected',
        'discarded_images': 'Discarded',
        'transmission_percent': 'Transmission %',
        'reduction_percent': 'Reduction %',
        'total_inference_time_per_image': 'Inference Time per Image (s)',
        'inference_per_second': 'Inference Per Second',
        'avg_batch_time_sec': 'Avg Batch Time (s)',
        'selected_score_mean': 'Mean Score',
        'selected_score_min': 'Min Score',
        'selected_score_max': 'Max Score',
    }
    columns_to_drop = ['output_csv', 'selected_images_dir', 'copy_stats']

    df_transmission_scenarios = pd.DataFrame(df_transmission_summary['transmission_scenarios'])
    df_transmission_scenarios.rename(columns=columns_to_rename, inplace=True)
    df_transmission_scenarios.drop(columns=columns_to_drop, inplace=True)

    df_transmission_scenarios.set_index('Scenario', inplace=True)
    df_transmission_scenarios.head()

    df_transmission_scenarios.to_csv(os.path.join(REPORT_DIR, 'transmission_comparison.csv'))


def display_images(source, title, n=6):
    if isinstance(source, str):
        all_images = [
            os.path.join(source, f)
            for f in os.listdir(source)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"))
        ]
        sample_n = min(n, len(all_images))
        image_paths = np.random.choice(all_images, sample_n, replace=False)
    else:
        sample_n = min(n, len(source))
        image_paths = np.random.choice(np.array(source), sample_n, replace=False)

    n_images = len(image_paths)

    if n_images == 0:
        print("No images to display.")
        return

    if n_images == 1:
        cols = 1
    elif n_images <= 4:
        cols = 2
    elif n_images <= 9:
        cols = 3
    else:
        cols = 3

    rows = math.ceil(n_images / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))

    if rows == 1 and cols == 1:
        axes = np.array([axes])
    else:
        axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for i, image_path in enumerate(image_paths):
        try:
            img = Image.open(image_path)
            axes[i].imshow(img)
            axes[i].axis("off")
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Błąd:\n{e}", ha="center", va="center")
            axes[i].axis("off")

    fig.suptitle(title, fontsize=22)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.suptitle(title + f'\n\n', fontsize=30)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, f'{title}.png'))
    plt.show()


generate_report()
hist_error()
transmission_comparision()
display_images(images['10 pct'], "10% Transmission")
display_images(images['5 pct'], "5% Transmission")
display_images(images['1 pct'], "1% Transmission")
display_images(lowest_rejected_images, "Rejected Images with Lowest Reconstruction Error")
display_images(random_rejected_images, "Randomly Selected Rejected Images")
# %%