import numpy as np
from PIL import Image

from svetovid import MarsAcceptableDataset, collect_image_paths, is_image_file


def _make_image(path, size=(320, 240)):
    array = (np.random.rand(size[1], size[0], 3) * 255).astype("uint8")
    Image.fromarray(array).save(path)


def test_is_image_file_filters_extensions():
    assert is_image_file("frame.JPG")
    assert is_image_file("scan.png")
    assert not is_image_file("notes.txt")
    assert not is_image_file("model.pth")


def test_collect_image_paths_recursive_and_sorted(tmp_path):
    (tmp_path / "nested").mkdir()
    _make_image(tmp_path / "b.jpg")
    _make_image(tmp_path / "nested" / "a.png")
    (tmp_path / "ignore.txt").write_text("not an image")

    paths = collect_image_paths(tmp_path)
    assert len(paths) == 2
    assert paths == sorted(paths)


def test_dataset_returns_normalized_grayscale_tensor(tmp_path):
    img_path = tmp_path / "sample.jpg"
    _make_image(img_path)

    dataset = MarsAcceptableDataset([str(img_path)], image_size=64)
    x, returned_path = dataset[0]

    assert len(dataset) == 1
    assert returned_path == str(img_path)
    assert x.shape == (1, 64, 64)          # grayscale, resized
    assert 0.0 <= x.min().item() <= x.max().item() <= 1.0
