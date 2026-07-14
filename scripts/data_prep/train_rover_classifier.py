import os 
import json
import copy
from pathlib import Path 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V3_Small_Weights
from PIL import ImageFile

import argparse
_p = argparse.ArgumentParser(description="Train MobileNetV3-Small binary classifier (acceptable vs rover_heavy)")
_p.add_argument("--data-dir", required=True, help="ImageFolder root with class subdirectories")
_args = _p.parse_args()


DATA_DIR = _args.data_dir
BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_CLASSES = 2
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
NUM_WORKERS = 0
MODEL_SAVE_PATH = "mobilenetv3_rover_classifier_best.pth"
CLASS_MAP_SAVE_PATH = "class_to_idx.json"

ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'using device: {device}')

train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels = 3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=8),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

full_dataset = datasets.ImageFolder(root=DATA_DIR)
print(f'Klasy: {full_dataset.classes}')
print(f'Liczba obrazów: {len(full_dataset)}')

with open(CLASS_MAP_SAVE_PATH, 'w', encoding='utf-8') as f:
    json.dump(full_dataset.class_to_idx, f, indent=4, ensure_ascii=False)

train_size = int(TRAIN_RATIO * len(full_dataset))
val_size = len(full_dataset) - train_size

generator = torch.Generator().manual_seed(RANDOM_SEED)
train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)


train_dataset = copy.copy(train_subset)
val_dataset = copy.copy(val_subset)

train_dataset.dataset = copy.copy(full_dataset)
val_dataset.dataset = copy.copy(full_dataset)

train_dataset.dataset.transform = train_transforms
val_dataset.dataset.transform = val_transforms

train_loader = DataLoader(
    train_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True,
    num_workers = NUM_WORKERS

)

val_loader = DataLoader(
    val_dataset,
    batch_size = BATCH_SIZE,
    shuffle=False,
    num_workers = NUM_WORKERS
)

dataloaders = {
    'train': train_loader,
    'val': val_loader
}
dataset_sizes = {
    "train": len(train_dataset),
    "val": len(val_dataset)
}
print(f'Train: {dataset_sizes["train"]}')
print(f'Val: {dataset_sizes["val"]}')

weights = MobileNet_V3_Small_Weights.DEFAULT
model = models.mobilenet_v3_small(weights=weights)

in_features = model.classifier[3].in_features

model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=10):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 40)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels).item()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print(f"{phase.upper()} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            if phase == 'val':
                scheduler.step(epoch_loss)

                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)
                    print(f"Zapisano nowy najlepszy model -> {MODEL_SAVE_PATH}")

    print('Trening complete')
    print(f'Best val acc: {best_val_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model

trained_model = train_model(
    model=model,
    dataloaders = dataloaders,
    dataset_sizes = dataset_sizes,
    criterion=criterion,
    optimizer = optimizer,
    scheduler=scheduler,
    num_epochs=NUM_EPOCHS

    )

print(f'\nDone.')