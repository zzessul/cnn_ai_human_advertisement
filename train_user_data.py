import argparse
import hashlib
import json
import random
import ssl
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import certifi
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


SEED = 42
CLASS_NAMES = ["인간광고", "AI광고"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def candidate_class_dirs(root: Path) -> list[tuple[Path, int]]:
    """명시적으로 AI/인간이라고 라벨된 폴더만 사용한다."""
    folders = []
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        normalized = unicodedata.normalize("NFC", path.name).replace(" ", "")
        if normalized in {"AI", "AI광고"}:
            folders.append((path, 1))
        elif normalized in {"인간", "인간광고", "캐릭터"}:
            folders.append((path, 0))
    return folders


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_images(root: Path) -> tuple[list[str], list[int], dict]:
    records = []
    skipped_invalid = []
    for folder, label in candidate_class_dirs(root):
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
                records.append((path, label, sha256(path)))
            except (OSError, UnidentifiedImageError):
                skipped_invalid.append(str(path))

    # 동일 파일이 여러 폴더에 복사돼 있어도 한 번만 학습한다.
    by_hash = {}
    conflicts = []
    for path, label, digest in records:
        if digest in by_hash and by_hash[digest][1] != label:
            conflicts.append([str(by_hash[digest][0]), str(path)])
            continue
        by_hash.setdefault(digest, (path, label))

    paths = [str(value[0]) for value in by_hash.values()]
    labels = [value[1] for value in by_hash.values()]
    audit = {
        "source_root": str(root.resolve()),
        "labeled_folders": [str(p) for p, _ in candidate_class_dirs(root)],
        "files_before_deduplication": len(records),
        "exact_duplicates_removed": len(records) - len(by_hash),
        "invalid_images_skipped": skipped_invalid,
        "cross_label_hash_conflicts": conflicts,
        "class_counts_after_deduplication": {
            CLASS_NAMES[i]: labels.count(i) for i in range(2)
        },
    }
    return paths, labels, audit


class ImageDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
        return self.transform(image), self.labels[index]


def make_transforms(size: int):
    train = transforms.Compose([
        transforms.Resize((size + 32, size + 32)),
        transforms.RandomResizedCrop(size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.12)),
    ])
    evaluate = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train, evaluate


def build_model(pretrained: bool = True):
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4), nn.Linear(in_features, 256), nn.SiLU(),
        nn.Dropout(0.2), nn.Linear(256, 2)
    )
    return model


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = correct = total = 0
    all_targets, all_predictions = [], []
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
        predictions = logits.argmax(1)
        loss_sum += loss.item() * targets.size(0)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)
        all_targets.extend(targets.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())
    return loss_sum / total, correct / total, all_targets, all_predictions


def split_data(paths, labels):
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=0.30, random_state=SEED, stratify=labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, random_state=SEED, stratify=temp_labels
    )
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/user_data"))
    parser.add_argument("--head-epochs", type=int, default=8)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    seed_everything(SEED)
    device = pick_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths, labels, audit = collect_images(args.data_dir)
    if min(Counter(labels).values(), default=0) < 10:
        raise ValueError(f"클래스별 이미지가 부족합니다: {Counter(labels)}")
    splits = split_data(paths, labels)
    audit["split_counts"] = {
        name: {CLASS_NAMES[i]: split_labels.count(i) for i in range(2)}
        for name, (_, split_labels) in zip(("train", "validation", "test"), splits)
    }
    (args.output_dir / "data_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit["class_counts_after_deduplication"], ensure_ascii=False))
    print(f"Device: {device}")

    train_tf, eval_tf = make_transforms(args.image_size)
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = splits
    counts = Counter(train_labels)
    sampler = WeightedRandomSampler(
        [1.0 / counts[label] for label in train_labels], len(train_labels), replacement=True
    )
    train_loader = DataLoader(ImageDataset(train_paths, train_labels, train_tf),
                              batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(ImageDataset(val_paths, val_labels, eval_tf),
                            batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(ImageDataset(test_paths, test_labels, eval_tf),
                             batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_accuracy = -1.0
    best_path = args.output_dir / "best_model.pth"

    phases = [
        ("head", args.head_epochs, 1e-3, True),
        ("finetune", args.finetune_epochs, 3e-5, False),
    ]
    history = []
    patience = 7
    for phase, epochs, lr, freeze in phases:
        stale = 0
        for name, parameter in model.named_parameters():
            parameter.requires_grad = ("classifier" in name) if freeze else True
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=1e-4
        )
        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
            va_loss, va_acc, _, _ = run_epoch(model, val_loader, criterion, device)
            row = {"phase": phase, "epoch": epoch, "train_loss": tr_loss,
                   "train_accuracy": tr_acc, "validation_loss": va_loss,
                   "validation_accuracy": va_acc}
            history.append(row)
            print(f"{phase} {epoch:02d}/{epochs} train={tr_acc:.3f} val={va_acc:.3f}")
            if va_acc > best_accuracy:
                best_accuracy, stale = va_acc, 0
                torch.save({
                    "model_state_dict": model.state_dict(), "class_names": CLASS_NAMES,
                    "image_size": args.image_size, "mean": MEAN, "std": STD,
                    "validation_accuracy": va_acc,
                }, best_path)
            else:
                stale += 1
            if phase == "finetune" and stale >= patience:
                print(f"Early stopping (patience={patience})")
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_accuracy, targets, predictions = run_epoch(
        model, test_loader, criterion, device
    )
    report = classification_report(
        targets, predictions, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    metrics = {
        "best_validation_accuracy": best_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "confusion_matrix": confusion_matrix(targets, predictions).tolist(),
        "classification_report": report,
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "test_manifest.json").write_text(
        json.dumps([{"path": p, "label": CLASS_NAMES[y]} for p, y in zip(test_paths, test_labels)],
                   ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Saved: {best_path}")


if __name__ == "__main__":
    main()
