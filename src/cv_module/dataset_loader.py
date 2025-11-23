import os
import random
from typing import List, Tuple
from PIL import Image

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

# ------------------------------------------------
# EXACT DATASET PATH (from your message)
# ------------------------------------------------
DATA_DIR = "data/raw/plantvillage/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"

# Crops you want to include
SELECTED_CROPS = ["tomato", "potato", "apple", "grape", "pepper"]

# Save final class list here (used by infer.py)
CLASS_OUTPUT_PATH = "models/disease_model/classes.txt"


# ------------------------------------------------
#   Transforms
# ------------------------------------------------
def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    return train_tf, val_tf


# ------------------------------------------------
#   Balanced sampling per class
# ------------------------------------------------
def balanced_reduce(class_to_paths: dict, reduce_ratio: float):
    reduced = []
    for cls, paths in class_to_paths.items():
        random.shuffle(paths)
        keep = max(1, int(len(paths) * reduce_ratio))
        reduced.extend(paths[:keep])
    random.shuffle(reduced)
    return reduced


# ------------------------------------------------
#   Custom Dataset
# ------------------------------------------------
class PathLabelDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


# ------------------------------------------------
#   MAIN LOADER
# ------------------------------------------------
def load_dataset(batch_size: int = 32,
                 val_split: float = 0.2,
                 reduce_ratio: float = 0.2):
    """
    Returns:
        train_dataset, val_dataset, selected_classes(list)
    """

    print("📂 Using dataset root:", DATA_DIR)

    # Load using ImageFolder to read classes
    full = datasets.ImageFolder(DATA_DIR)
    all_classes = full.classes

    print("Total classes detected:", len(all_classes))

    # Filter based on crop keywords
    selected_classes = [
        cls for cls in all_classes
        if any(crop in cls.lower() for crop in SELECTED_CROPS)
    ]

    if not selected_classes:
        raise ValueError(f"Could not match classes using crops {SELECTED_CROPS}")

    print("\n🌱 Selected classes:")
    for c in selected_classes:
        print(" -", c)

    # Build mapping: class → image paths
    class_to_paths = {cls: [] for cls in selected_classes}

    for path, idx in full.samples:
        cls_name = full.classes[idx]
        if cls_name in class_to_paths:
            class_to_paths[cls_name].append(path)

    print("\n📊 Sample counts BEFORE reduction:")
    for c, lst in class_to_paths.items():
        print(f"{c}: {len(lst)}")

    # Balanced reduction
    final_paths = balanced_reduce(class_to_paths, reduce_ratio)

    print("\n📊 Sample counts AFTER reduction (approx):")
    per_class = {c: 0 for c in selected_classes}
    for p in final_paths:
        per_class[p.split("/")[-2]] += 1
    for c, cnt in per_class.items():
        print(f"{c}: {cnt}")

    # Build label mapping
    class_to_idx = {cls: i for i, cls in enumerate(selected_classes)}

    labelled = [(path, class_to_idx[path.split("/")[-2]]) for path in final_paths]

    # Stratified split
    train_samples = []
    val_samples = []
    per_class_split = {cls: [] for cls in selected_classes}

    for path, idx in labelled:
        cls = selected_classes[idx]
        per_class_split[cls].append((path, idx))

    for cls, items in per_class_split.items():
        n = len(items)
        split_at = int((1 - val_split) * n)
        split_at = max(1, min(split_at, n - 1))
        train_samples.extend(items[:split_at])
        val_samples.extend(items[split_at:])

    print("\n📦 Final train size:", len(train_samples))
    print("📦 Final val size:", len(val_samples))

    # Save classes.txt
    os.makedirs(os.path.dirname(CLASS_OUTPUT_PATH), exist_ok=True)
    with open(CLASS_OUTPUT_PATH, "w") as f:
        for cls in selected_classes:
            f.write(cls + "\n")

    print("\n📝 Saved class list to:", CLASS_OUTPUT_PATH)

    # Create Dataset objects
    train_tf, val_tf = get_transforms()
    train_dataset = PathLabelDataset(train_samples, train_tf)
    val_dataset = PathLabelDataset(val_samples, val_tf)

    return train_dataset, val_dataset, selected_classes


# DEBUG
if __name__ == "__main__":
    td, vd, classes = load_dataset()
    print("\nLoaded dataset:")
    print("Train samples:", len(td))
    print("Val samples:", len(vd))
    print("Classes:", classes)
