import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4), nn.Linear(in_features, 256), nn.SiLU(),
        nn.Dropout(0.2), nn.Linear(256, 2)
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path("artifacts/user_data/best_model.pth"))
    args = parser.parse_args()
    device = pick_device()
    checkpoint = torch.load(args.model, map_location=device, weights_only=True)
    tf = transforms.Compose([
        transforms.Resize((checkpoint["image_size"], checkpoint["image_size"])),
        transforms.ToTensor(), transforms.Normalize(checkpoint["mean"], checkpoint["std"]),
    ])
    model = build_model().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with Image.open(args.image) as image:
        tensor = tf(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().tolist()
    winner = max(range(len(probabilities)), key=probabilities.__getitem__)
    print(f"예측: {checkpoint['class_names'][winner]} ({probabilities[winner] * 100:.1f}%)")
    for name, probability in zip(checkpoint["class_names"], probabilities):
        print(f"  {name}: {probability * 100:.1f}%")


if __name__ == "__main__":
    main()
