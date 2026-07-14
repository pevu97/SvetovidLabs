"""Dataset utilities shared by training and inference."""

import os

from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

ImageFile.LOAD_TRUNCATED_IMAGES = True


def is_image_file(filename: str) -> bool:
    return filename.lower().endswith(IMAGE_EXTENSIONS)


def collect_image_paths(root_dir) -> list:
    """Recursively collect and sort all image paths under root_dir."""
    paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if is_image_file(file):
                paths.append(os.path.join(root, file))
    paths.sort()
    return paths


class MarsAcceptableDataset(Dataset):
    """Grayscale, resized NAVCAM images as tensors in [0, 1].

    With augment=True (training only) applies light geometric augmentation:
    random horizontal flip and up to 5 degrees of rotation.
    """

    def __init__(self, image_paths, image_size: int = 256, augment: bool = False):
        self.image_paths = image_paths
        self.augment = augment

        if augment:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                x = self.transform(img)
        except Exception as e:
            raise RuntimeError(f"Error loading image: {path} | {e}")

        return x, path
