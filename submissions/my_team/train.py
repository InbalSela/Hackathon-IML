from pathlib import Path
from collections import defaultdict
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
AUG_ROOT  = Path(__file__).resolve().parents[2] / "augmentations"
OUTPUT    = Path(__file__).resolve().parent / "weights.joblib"

IMAGE_SIZE = 224
BATCH_SIZE = 64
EPOCHS     = 50
LR         = 1e-3
WEIGHT_DECAY = 1e-4

TRAIN_FRACTION = 0.5  # 500 images/class
VAL_FRACTION   = 0.4  # 400 images/class
TEST_FRACTION  = 0.1  # 100 images/class — DO NOT touch until final run

# ImageNet channel mean and std — must match what evaluate.py uses
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


class GaussianNoise:
    """
    Adds random Gaussian noise to a tensor (applied between ToTensor and Normalize).
    Simulates sensor noise and image compression artifacts.
    """
    def __init__(self, std: float = 0.05, p: float = 0.3):
        self.std = std
        self.p   = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            return (tensor + torch.randn_like(tensor) * self.std).clamp(0.0, 1.0)
        return tensor


def build_transforms(train: bool) -> transforms.Compose:
    """
    Training pipeline — rich augmentations to improve robustness:
      - RandomResizedCrop:  simulates close-up / zoom-in on object parts
      - RandomHorizontalFlip + RandomVerticalFlip: mirroring
      - RandomRotation:     rotation up to 30 degrees
      - RandomAffine:       stretch and shear (densed/stretched pictures)
      - ColorJitter:        brightness, contrast, saturation, hue changes
      - RandomGrayscale:    forces model to learn shape, not just color
      - RandomAutocontrast: simulates different contrast levels
      - GaussianNoise:      simulates sensor noise
      - RandomErasing:      simulates occlusion (part of object hidden)

    Validation/Test pipeline — deterministic, no randomness:
      - Pad + Resize + CenterCrop only
    """
    if train:
        return transforms.Compose([
            # Pad small images to avoid distortion, then zoom-in crop
            transforms.Pad(padding=IMAGE_SIZE // 2, fill=0),
            # RandomResizedCrop: randomly zooms into 50-100% of the image
            # teaches the model to recognize partial/close-up views
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(degrees=30),
            # Stretch/shear: simulates densed or stretched pictures
            transforms.RandomAffine(degrees=0, shear=15, scale=(0.8, 1.2)),
            # Color changes: brightness, contrast, saturation, hue
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            # Black-and-white: forces the model to use shape, not color
            transforms.RandomGrayscale(p=0.2),
            # Contrast variation
            transforms.RandomAutocontrast(p=0.3),
            transforms.ToTensor(),
            # Gaussian noise (applied before normalize so values stay in [0,1])
            GaussianNoise(std=0.05, p=0.3),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            # RandomErasing: blacks out a random rectangle — simulates occlusion
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
        ])
    else:
        return transforms.Compose([
            transforms.Pad(padding=IMAGE_SIZE // 2, fill=0),
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


class SplitDataset(torch.utils.data.Dataset):
    """Dataset built from an explicit list of (img_path, label) pairs."""

    def __init__(self, samples, transform):
        self.samples   = samples
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
      50% training   (500 images/class = 10,000 total)
      40% validation (400 images/class =  8,000 total)
      10% test       (100 images/class =  2,000 total) — held out until final run

    Fixed seed guarantees the exact same splits on every run.
    """
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(full_dataset.samples):
        class_indices[label].append(idx)

    rng = torch.Generator().manual_seed(seed)
    train_samples, val_samples, test_samples = [], [], []

    for label, indices in sorted(class_indices.items()):
        n_train = int(len(indices) * TRAIN_FRACTION)  # 500
        n_val   = int(len(indices) * VAL_FRACTION)    # 400
        # remaining 100 go to test

        perm     = torch.randperm(len(indices), generator=rng).tolist()
        shuffled = [indices[i] for i in perm]

        for idx in shuffled[:n_train]:
            train_samples.append(full_dataset.samples[idx])
        for idx in shuffled[n_train:n_train + n_val]:
            val_samples.append(full_dataset.samples[idx])
        for idx in shuffled[n_train + n_val:]:
            test_samples.append(full_dataset.samples[idx])

    return train_samples, val_samples, test_samples


def load_augmentation_samples():
    """
    Load pre-saved augmented images from the augmentations/ folder.
    These are added to the training set as additional data.

    Structure expected:
      augmentations/
        color_jitter/
          acoustic_guitar/  (*.jpg)
          ...
        random_rotation/
          acoustic_guitar/
          ...
    """
    from labels import HF_INDEX_TO_NAME, HF_INDEX_TO_IDX

    # Map class name -> local index (0-19)
    name_to_idx = {
        name: HF_INDEX_TO_IDX[hf_idx]
        for hf_idx, name in HF_INDEX_TO_NAME.items()
    }

    samples = []

    if not AUG_ROOT.exists():
        print(f"Warning: {AUG_ROOT} not found — skipping pre-saved augmentations")
        return samples

    for aug_dir in sorted(AUG_ROOT.iterdir()):
        if not aug_dir.is_dir():
            continue
        for class_dir in sorted(aug_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            if class_name not in name_to_idx:
                continue
            label = name_to_idx[class_name]
            for ext in ("*.jpg", "*.jpeg", "*.JPEG", "*.png"):
                for img_path in sorted(class_dir.glob(ext)):
                    samples.append((img_path, label))

    print(f"Loaded {len(samples)} pre-saved augmentation images from {AUG_ROOT.name}/")
    return samples


def get_dataloaders():
    full_dataset = ImageNetSubset(DATA_ROOT, split="train", transform=None)

    # Three-way stratified split — same seed = same splits every run
    train_samples, val_samples, test_samples = stratified_split(full_dataset)

    # Add pre-saved augmentation images to training set only
    aug_samples = load_augmentation_samples()
    train_samples = train_samples + aug_samples

    print(
        f"Train: {len(train_samples)} (incl. {len(aug_samples)} pre-saved aug) | "
        f"Val: {len(val_samples)} | "
        f"Test (held-out): {len(test_samples)} — DO NOT evaluate until final run"
    )

    train_dataset = SplitDataset(train_samples, transform=build_transforms(train=True))
    val_dataset   = SplitDataset(val_samples,   transform=build_transforms(train=False))

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=pin)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=pin)

    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """One full pass over the training set. Returns avg loss and accuracy."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    """Run inference on a loader, return accuracy."""
    model.eval()
    correct, total = 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds    = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return correct / total


def main():
    start = time.time()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Load dataset — 50/40/10 stratified split + pre-saved augmentations in training
    train_loader, val_loader = get_dataloaders()

    model   = ModelArchitecture(num_classes=20, dropout=0.5).to(device)
    # Label smoothing: prevents overconfidence, improves generalization
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    best_state   = None

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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  -> New best val acc: {best_val_acc:.4f} — checkpoint saved")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Total time: {(time.time() - start) / 60:.1f} min")

    # Move to CPU before saving — required for hardware-independent loading
    joblib.dump(best_state, OUTPUT)
    print(f"Saved weights to {OUTPUT}")


if __name__ == "__main__":
    main()
