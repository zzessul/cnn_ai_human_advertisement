import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split

# ── 재현성 ────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ── 설정 ──────────────────────────────────────────────────────────────────
DATA_DIR      = "광고_최종"
CLASS_NAMES   = ["인간광고", "AI광고"]   # 0=인간, 1=AI
IMG_SIZE      = 224
BATCH_SIZE    = 16
PHASE1_EPOCHS = 10    # backbone 고정, head만 학습
PHASE2_EPOCHS = 40    # 전체 fine-tune
LR_HEAD       = 1e-3
LR_FINETUNE   = 3e-5
WEIGHT_DECAY  = 1e-4
VAL_RATIO     = 0.2
SAVE_PATH     = "best_model.pth"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")

# ── 데이터셋 ──────────────────────────────────────────────────────────────
def collect_image_paths(data_dir, class_names):
    """클래스 폴더에서 이미지 경로와 레이블을 수집."""
    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths, labels = [], []
    for label, cls in enumerate(class_names):
        folder = os.path.join(data_dir, cls)
        for fname in os.listdir(folder):
            if os.path.splitext(fname)[1].lower() in EXTS:
                paths.append(os.path.join(folder, fname))
                labels.append(label)
    return paths, labels


class AdDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ── Transform ─────────────────────────────────────────────────────────────
# ImageNet 평균/표준편차
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomRotation(15),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ── 모델 ──────────────────────────────────────────────────────────────────
def build_model(num_classes=2, dropout=0.4):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    # classifier 교체
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 256),
        nn.SiLU(),
        nn.Dropout(dropout / 2),
        nn.Linear(256, num_classes),
    )
    return model


def freeze_backbone(model):
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


# ── 학습/평가 루프 ─────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None, phase="train"):
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss    = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)

    return total_loss / total, correct / total


# ── 메인 ──────────────────────────────────────────────────────────────────
def main():
    # 데이터 로드 & 분할
    all_paths, all_labels = collect_image_paths(DATA_DIR, CLASS_NAMES)
    print(f"전체 이미지: {len(all_paths)}  (인간:{all_labels.count(0)}, AI:{all_labels.count(1)})")

    tr_paths, val_paths, tr_labels, val_labels = train_test_split(
        all_paths, all_labels, test_size=VAL_RATIO, random_state=SEED, stratify=all_labels
    )
    print(f"Train: {len(tr_paths)}, Val: {len(val_paths)}")

    # 클래스 불균형 대응 (WeightedRandomSampler)
    class_counts = [tr_labels.count(i) for i in range(len(CLASS_NAMES))]
    sample_weights = [1.0 / class_counts[l] for l in tr_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(tr_labels), replacement=True)

    train_ds = AdDataset(tr_paths,  tr_labels,  train_tf)
    val_ds   = AdDataset(val_paths, val_labels, val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 모델
    model     = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Phase 1: head only ────────────────────────────────────────────────
    print("\n=== Phase 1: Classifier Head만 학습 ===")
    freeze_backbone(model)
    optimizer1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=WEIGHT_DECAY
    )
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=PHASE1_EPOCHS)

    best_val_acc = 0.0
    for epoch in range(1, PHASE1_EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer1, "train")
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, None,       "val")
        scheduler1.step()

        marker = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), SAVE_PATH)
            marker = "  ← best"

        print(f"[P1 {epoch:02d}/{PHASE1_EPOCHS}] "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val loss={vl_loss:.4f} acc={vl_acc:.4f}{marker}")

    # ── Phase 2: 전체 fine-tune ───────────────────────────────────────────
    print("\n=== Phase 2: 전체 모델 Fine-tune ===")
    unfreeze_all(model)
    optimizer2 = optim.AdamW(model.parameters(), lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=PHASE2_EPOCHS, eta_min=1e-6)

    patience, no_improve = 12, 0

    for epoch in range(1, PHASE2_EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer2, "train")
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, None,       "val")
        scheduler2.step()

        marker = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), SAVE_PATH)
            marker = "  ← best"
            no_improve = 0
        else:
            no_improve += 1

        print(f"[P2 {epoch:02d}/{PHASE2_EPOCHS}] "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val loss={vl_loss:.4f} acc={vl_acc:.4f}{marker}")

        if no_improve >= patience:
            print(f"Early stopping (patience={patience})")
            break

    print(f"\n최고 Val Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.1f}%)")
    print(f"모델 저장 완료: {SAVE_PATH}")


if __name__ == "__main__":
    main()
