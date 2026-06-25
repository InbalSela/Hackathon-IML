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
TRAIN_FRACTION = 0.8   # 800 images/class = 16,000 total
VAL_FRACTION   = 0.1   # 100 images/class =  2,000 total
TEST_FRACTION  = 0.1   # 100 images/class =  2,000 total — held out until final run
FINAL_RUN      = True # set True to train on 90% and evaluate test set once at end
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4

# ImageNet channel mean and std — must match what evaluate.py uses
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def build_transforms(train: bool) -> transforms.Compose:
    """
    Training pipeline: augmentations to improve robustness and reduce overfitting.
    Validation pipeline: deterministic, no randomness.
    """
    if train:
        return transforms.Compose([
            transforms.Pad(padding=IMAGE_SIZE // 2, fill=0),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
            transforms.RandomGrayscale(p=0.15),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
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
      - 80% training   (800 images/class = 16,000 total)
      - 10% validation (100 images/class =  2,000 total)  <- used every run
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

    if FINAL_RUN:
        # Merge val into train — use all 90%, evaluate on test once at end
        train_samples = train_samples + val_samples
        val_samples = []
        print(
            f"FINAL RUN — Train: {len(train_samples)} | "
            f"Test: {len(test_samples)} (evaluated once at end)"
        )
    else:
        print(
            f"Train: {len(train_samples)} | "
            f"Val: {len(val_samples)} | "
            f"Test (held-out): {len(test_samples)} — DO NOT evaluate until final run"
        )

    pin = torch.cuda.is_available()

    train_dataset = SplitDataset(train_samples, transform=build_transforms(train=True))
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=pin)

    if FINAL_RUN:
        test_dataset = SplitDataset(test_samples, transform=build_transforms(train=False))
        test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=pin)
        return train_loader, None, test_loader

    val_dataset  = SplitDataset(val_samples, transform=build_transforms(train=False))
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=pin)
    return train_loader, val_loader, None


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

    train_loader, val_loader, test_loader = get_dataloaders()

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
        scheduler.step()

        if FINAL_RUN:
            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )
        else:
            val_acc = evaluate(model, val_loader, device)
            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f"  -> New best val acc: {best_val_acc:.4f} — checkpoint saved")

    if FINAL_RUN:
        # Save final epoch weights and evaluate on test set once
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        test_acc = evaluate(model, test_loader, device)
        print(f"\nFinal run complete. Test accuracy: {test_acc:.4f}")
    else:
        print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Total time: {(time.time() - start) / 60:.1f} min")

    # Save best weights to weights.joblib (required by the submission format)
    joblib.dump(best_state, OUTPUT)
    print(f"Saved weights to {OUTPUT}")


if __name__ == "__main__":
    main()