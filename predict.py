import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

CLASS_NAMES = ["인간광고", "AI광고"]
IMG_SIZE    = 224
MEAN        = [0.485, 0.456, 0.406]
STD         = [0.229, 0.224, 0.225]
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def build_model(num_classes=2, dropout=0.4):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 256),
        nn.SiLU(),
        nn.Dropout(dropout / 2),
        nn.Linear(256, num_classes),
    )
    return model


def predict(image_path: str, model_path: str = "best_model.pth") -> dict:
    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    img    = Image.open(image_path).convert("RGB")
    tensor = tf(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    pred_idx  = probs.argmax().item()
    pred_cls  = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx].item()

    print(f"이미지: {image_path}")
    print(f"예측:   {pred_cls}  ({confidence * 100:.1f}%)")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"  {cls}: {probs[i].item() * 100:.1f}%")

    return {"class": pred_cls, "confidence": confidence, "probs": probs.tolist()}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python predict.py <이미지경로> [모델경로]")
        print("예시:   python predict.py 광고_최종/AI광고/AI_01_가방.png")
        sys.exit(1)

    img_path   = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "best_model.pth"
    predict(img_path, model_path)
