import json

import torch

from svetovid import compute_batch_reconstruction_errors, save_json, set_seed


def test_error_is_zero_for_identical_tensors():
    x = torch.rand(4, 1, 16, 16)
    errors = compute_batch_reconstruction_errors(x, x.clone())
    assert errors.shape == (4,)
    assert torch.allclose(errors, torch.zeros(4), atol=1e-7)


def test_error_is_one_for_opposite_extremes():
    zeros = torch.zeros(3, 1, 8, 8)
    ones = torch.ones(3, 1, 8, 8)
    errors = compute_batch_reconstruction_errors(zeros, ones)
    assert torch.allclose(errors, torch.ones(3))


def test_error_orders_by_severity():
    """A worse reconstruction must yield a strictly higher error."""
    target = torch.zeros(1, 1, 8, 8)
    slightly_off = torch.full((1, 1, 8, 8), 0.1)
    badly_off = torch.full((1, 1, 8, 8), 0.7)

    small = compute_batch_reconstruction_errors(target, slightly_off)
    large = compute_batch_reconstruction_errors(target, badly_off)
    assert small.item() < large.item()


def test_set_seed_makes_torch_deterministic():
    set_seed(123)
    a = torch.rand(5)
    set_seed(123)
    b = torch.rand(5)
    assert torch.equal(a, b)


def test_save_json_writes_valid_utf8(tmp_path):
    path = tmp_path / "out.json"
    save_json({"file": "łazik.jpg", "error": 0.5}, path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["file"] == "łazik.jpg"
