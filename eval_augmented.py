"""
Evaluate saved weights against the provided augmented test sets.

Run from the hackathon root:
    python eval_augmented.py
"""
import sys
import importlib.util
from pathlib import Path

import torch
import joblib
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from labels import HF_INDEX_TO_NAME, HF_INDEX_TO_IDX, TARGET_HF_INDICES

HACKATHON_ROOT  = Path(__file__).resolve().parent
AUG_ROOT        = HACKATHON_ROOT / "augmentations"
TEAM_DIR        = HACKATHON_ROOT / "submissions" / "my_team"
WEIGHTS_PATH    = TEAM_DIR / "weights.joblib"
BATCH_SIZE      = 64

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Map class folder name → local index
NAME_TO_IDX = {
    HF_INDEX_TO_NAME[hf_idx]: HF_INDEX_TO_IDX[hf_idx]
    for hf_idx in TARGET_HF_INDICES
}


class AugDataset(Dataset):
    def __init__(self, aug_dir: Path):
        self.samples = []
        for class_name, local_idx in NAME_TO_IDX.items():
            class_dir = aug_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in sorted(class_dir.glob("*.jpg")):
                self.samples.append((img_path, local_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return TRANSFORM(img), label


def load_model():
    sys.path.insert(0, str(TEAM_DIR))
    sys.modules.pop("model", None)

    spec = importlib.util.spec_from_file_location("predict", TEAM_DIR / "predict.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.Model()
    model.load(str(WEIGHTS_PATH))

    sys.path.pop(0)
    sys.modules.pop("model", None)
    return model


@torch.no_grad()
def evaluate(model, loader):
    correct, total = 0, 0
    for x, y in loader:
        preds = model.predict(x)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total


def main():
    print(f"Loading model from {WEIGHTS_PATH}\n")
    model = load_model()

    aug_types = sorted(d.name for d in AUG_ROOT.iterdir() if d.is_dir())
    results = {}

    for aug_name in aug_types:
        dataset = AugDataset(AUG_ROOT / aug_name)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        acc = evaluate(model, loader)
        results[aug_name] = acc
        print(f"{aug_name:<25} {len(dataset):>4} images   acc: {acc:.4f} ({acc*100:.1f}%)")

    overall = sum(results.values()) / len(results)
    print(f"\nOverall augmented acc: {overall:.4f} ({overall*100:.1f}%)")


if __name__ == "__main__":
    main()
