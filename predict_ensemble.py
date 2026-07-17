import argparse
import ssl
from pathlib import Path

import certifi
import joblib
import numpy as np
import torch

from train_advanced_ensemble import Images, device, load_encoder
from torch.utils.data import DataLoader


def encode_image(image_path, transform, encode, target_device):
    views = []
    with torch.inference_mode():
        for flip in (False, True):
            loader = DataLoader(Images([image_path], transform, flip), batch_size=1)
            features = encode(next(iter(loader)).to(target_device))
            if isinstance(features, (tuple, list)):
                features = features[0]
            if features.ndim > 2:
                features = features.mean(dim=tuple(range(2, features.ndim)))
            features = torch.nn.functional.normalize(features, dim=1)
            views.append(features.cpu().float().numpy())
    combined = views[0] + views[1]
    return combined / np.linalg.norm(combined, axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description="AI 티 점수(0~100) 추론")
    parser.add_argument("image", type=Path)
    parser.add_argument("--ensemble", type=Path,
                        default=Path("artifacts/advanced_ensemble/best_ensemble.joblib"))
    args = parser.parse_args()
    artifact = joblib.load(args.ensemble)
    target_device = device()
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

    probabilities = np.zeros(2, dtype=np.float64)
    for name, classifier in artifact["models"].items():
        model, transform, encode = load_encoder(artifact["encoders"][name], target_device)
        features = encode_image(args.image, transform, encode, target_device)
        model_probability = classifier.predict_proba(features)[0]
        probabilities += artifact["weights"][name] * model_probability
        del model
        if target_device.type == "mps":
            torch.mps.empty_cache()

    ai_score = float(probabilities[1] * 100)
    prediction = int(probabilities.argmax())
    print(f"예측: {artifact['class_names'][prediction]}")
    print(f"AI 티 점수: {ai_score:.1f}/100")
    print(f"인간광고 유사도: {probabilities[0] * 100:.1f}%")
    print(f"AI광고 유사도: {probabilities[1] * 100:.1f}%")
    print("주의: 이 점수는 현재 학습 데이터 기준의 모델 확률이며 사람 평가의 절대 점수가 아닙니다.")


if __name__ == "__main__":
    main()
