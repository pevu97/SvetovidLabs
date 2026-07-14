"""Shared helpers: reproducibility, metrics, serialization."""

import json
import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_batch_reconstruction_errors(inputs, outputs):
    """Per-image mean absolute error (L1) between input and reconstruction.

    Returns a 1-D tensor of shape (batch_size,).
    """
    return torch.mean(torch.abs(inputs - outputs), dim=(1, 2, 3))


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)
