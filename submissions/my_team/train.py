from pathlib import Path
import sys

import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import time

# Allow imports from the hackathon root (base_model.py, labels.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from base_model import ImageNetSubset
from model import ModelArchitecture


DATA_ROOT = Path(__file__).resolve().parents[2] / "dataset"
OUTPUT = Path(__file__).resolve().parent / "weights.joblib"

IMAGE_SIZE = 224   # standard ImageNet input size
BATCH_SIZE = 64
TRAIN_FRACTION = 0.5  # 50% training, 40% validation, 10% held-out test (per class)
VAL_FRACTION   = 0.4
TEST_FRACTION  = 0.1
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4

# ImageNet channel mean and std — must match what evaluate.py uses
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def build_transforms(train: bool) -> transforms.Compose:
    """
    Two separate pipelines:
      - train: augmentations (horizontal flip) to help generalization
      - val:   deterministic, no randomness, for reliable accuracy measurement

    Both pipelines:
      1. Pad images smaller than IMAGE_SIZE to avoid upscale distortion
      2. Resize shorter edge to 256
      3. CenterCrop to IMAGE_SIZE x IMAGE_SIZE
    """
    shared = [
        # If image is smaller than IMAGE_SIZE, pad with zeros (black borders)
        # instead of stretching/distorting the image content
        transforms.Pad(padding=IMAGE_SIZE // 2, fill=0),
        # Resize shorter edge to 256 — image is now at least 256 in both dimensions
        transforms.Resize(256),
        # Crop the center 224x224 region
        transforms.CenterCrop(IMAGE_SIZE),
    ]

    if train:
        return transforms.Compose([
            *shared,
            transforms.RandomHorizontalFlip(),
            # TODO: add more augmentations here to improve robustness (Phase 2)
            transforms.ToTensor(),
            # Normalize to ~[-2, 2] so the network gets zero-centered inputs
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            *shared,
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


class SplitDataset(torch.utils.data.Dataset):
    """A dataset built from an explicit list of (img_path, label) samples."""

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), label


def stratified_split(full_dataset, seed: int = 42):
    """
    Split each class into exactly:
      - 50% training   (500 images/class = 10,000 total)
      - 40% validation (400 images/class =  8,000 total)  <- used every run
      - 10% test       (100 images/class =  2,000 total)  <- never touch until final

    The same seed guarantees the exact same three splits on every run,
    so the test set is never accidentally exposed during development.
    """
    from collections import defaultdict

    # Group sample indices by class label
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(full_dataset.samples):
        class_indices[label].append(idx)

    rng = torch.Generator().manual_seed(seed)
    train_samples, val_samples, test_samples = [], [], []

    for label, indices in sorted(class_indices.items()):
        n_train = int(len(indices) * TRAIN_FRACTION)  # 500
        n_val   = int(len(indices) * VAL_FRACTION)    # 400
        # remaining goes to test (100)

        # Shuffle indices for this class before splitting
        perm = torch.randperm(len(indices), generator=rng).tolist()
        shuffled = [indices[i] for i in perm]

        for idx in shuffled[:n_train]:
            train_samples.append(full_dataset.samples[idx])
        for idx in shuffled[n_train:n_train + n_val]:
            val_samples.append(full_dataset.samples[idx])
        for idx in shuffled[n_train + n_val:]:
            test_samples.append(full_dataset.samples[idx])

    return train_samples, val_samples, test_samples


def get_dataloaders():
    # Load full dataset without transforms — each split applies its own
    full_dataset = ImageNetSubset(DATA_ROOT, split="train", transform=None)

    # Three-way stratified split — same seed = same splits every run
    train_samples, val_samples, test_samples = stratified_split(full_dataset)
    print(
        f"Train: {len(train_samples)} | "
        f"Val: {len(val_samples)} | "
        f"Test (held-out): {len(test_samples)} — DO NOT evaluate until final run"
    )

    train_dataset = SplitDataset(train_samples, transform=build_transforms(train=True))
    val_dataset   = SplitDataset(val_samples,   transform=build_transforms(train=False))
    # test_dataset is intentionally not loaded into a dataloader here

    # pin_memory speeds up CPU->GPU transfer but is not supported on MPS
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=pin)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=pin)

    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """Run one full pass over the training set, return loss and accuracy."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    """Run inference on the validation set, return accuracy."""
    model.eval()
    correct, total = 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


def main():
    """
    Full training pipeline.

    This script must create weights.joblib.
    """
    start = time.time()

    # Detect best available device (MPS for Apple Silicon, CUDA for NVIDIA, else CPU)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Load dataset with stratified 50/40/10 split (train/val/held-out test)
    train_loader, val_loader = get_dataloaders()

    # Create model and move to device
    model = ModelArchitecture(num_classes=20).to(device)

    # CrossEntropyLoss expects raw logits — handles softmax internally
    loss_fn = nn.CrossEntropyLoss()

    # Adam optimizer: adapts learning rate per parameter, works well from scratch
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Cosine annealing: smoothly decays LR from LR to near 0 over all epochs
    # avoids the model getting stuck near a sharp local minimum late in training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Track best validation accuracy to save the best checkpoint
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        # Save the weights that achieved the best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Move to CPU before saving so weights load correctly on any machine
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  -> New best val acc: {best_val_acc:.4f} — checkpoint saved")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Total time: {(time.time() - start) / 60:.1f} min")

    # Save best weights to weights.joblib (required by the submission format)
    joblib.dump(best_state, OUTPUT)
    print(f"Saved weights to {OUTPUT}")


if __name__ == "__main__":
    main()