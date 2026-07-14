
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Data
DATA_DIR = PROJECT_ROOT / "data"

# Inference results
RESULTS_DIR = PROJECT_ROOT / "inference results"
INFERENCE_RECORDS_PATH = RESULTS_DIR / "inference_records.json"
INFERENCE_SUMMARY_PATH = RESULTS_DIR / "inference_summary.json"


# Simulation results
SIMULATION_RESULTS_DIR = PROJECT_ROOT / "simulation results"
TRANSMISSION_SUMMARY_PATH = SIMULATION_RESULTS_DIR / "transmission_summary.json"

# Report
REPORT_DIR = PROJECT_ROOT / "report"


# Model
MODEL_PATH = PROJECT_ROOT / "best_autoencoder.pth"

