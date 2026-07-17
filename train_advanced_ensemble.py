import argparse
import itertools
import json
import ssl
from pathlib import Path

import certifi
import joblib
import numpy as np
import open_clip
import timm
import torch
from PIL import Image, ImageOps
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
from torchvision.transforms import v2

from dataset_utils import CLASS_NAMES, collect_images, split_data


ENCODERS = {
    "clip_l14": ("open_clip", "ViT-L-14", "openai"),
    "siglip_b16": ("open_clip", "ViT-B-16-SigLIP", "webli"),
    "dinov2_b14": ("torch_hub", "dinov2_vitb14", None),
    "convnextv2_b": ("timm", "convnextv2_base.fcmae_ft_in22k_in1k", None),
    "convnext_tiny": ("torchvision", "convnext_tiny", None),
}


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Images(Dataset):
    def __init__(self, paths, transform, flip=False):
        self.paths, self.transform, self.flip = paths, transform, flip

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            if self.flip:
                image = ImageOps.mirror(image)
            return self.transform(image)


def load_encoder(spec, target_device):
    family, model_name, pretrained = spec
    if family == "open_clip":
        model, _, transform = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=target_device
        )
        encode = model.encode_image
    elif family == "timm":
        model = timm.create_model(model_name, pretrained=True, num_classes=0).to(target_device)
        transform = create_transform(**resolve_model_data_config(model), is_training=False)
        encode = model.forward
    elif family == "torch_hub":
        model = torch.hub.load("facebookresearch/dinov2", model_name).to(target_device)
        transform = v2.Compose([
            v2.Resize(256, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(224), v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        encode = model.forward
    else:
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = convnext_tiny(weights=weights).to(target_device)
        model.classifier[-1] = torch.nn.Identity()
        transform = weights.transforms()
        encode = model.forward
    model.eval()
    return model, transform, encode


def extract(paths, transform, encode, target_device, batch_size, flip=False):
    loader = DataLoader(Images(paths, transform, flip), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    output = []
    with torch.inference_mode():
        for tensors in loader:
            features = encode(tensors.to(target_device))
            if isinstance(features, (tuple, list)):
                features = features[0]
            if features.ndim > 2:
                features = features.mean(dim=tuple(range(2, features.ndim)))
            features = torch.nn.functional.normalize(features, dim=1)
            output.append(features.cpu().float().numpy())
    return np.concatenate(output)


def embeddings(name, spec, splits, output_dir, target_device, batch_size):
    cache = output_dir / f"{name}_embeddings.npz"
    if cache.exists():
        values = np.load(cache)
        print(f"{name}: cached embeddings")
        return {split: values[split] for split in ("train", "validation", "test")}
    model, transform, encode = load_encoder(spec, target_device)
    result = {}
    for split, (paths, _) in splits.items():
        original = extract(paths, transform, encode, target_device, batch_size)
        flipped = extract(paths, transform, encode, target_device, batch_size, True)
        combined = original + flipped
        result[split] = combined / np.linalg.norm(combined, axis=1, keepdims=True)
        print(f"{name}: {split} {result[split].shape}")
    np.savez_compressed(cache, **result)
    del model
    if target_device.type == "mps":
        torch.mps.empty_cache()
    return result


def classifier(c):
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=c, max_iter=5000, class_weight="balanced",
                                     random_state=42)),
    ])


def choose_classifier(features, labels, cv):
    candidates = []
    for c in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0):
        scores = cross_val_score(classifier(c), features, labels, cv=cv, scoring="accuracy")
        candidates.append((float(scores.mean()), float(scores.std()), c))
    return max(candidates, key=lambda row: (row[0], -row[2])), candidates


def ensemble_candidates(names):
    for size in range(1, len(names) + 1):
        for subset in itertools.combinations(names, size):
            yield {name: 1 / size for name in subset}
    for first, second in itertools.combinations(names, 2):
        for first_weight in np.arange(0.1, 1.0, 0.1):
            yield {first: float(first_weight), second: float(1 - first_weight)}


def weighted_probabilities(probabilities, weights):
    return sum(probabilities[name] * weight for name, weight in weights.items())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--encoders", nargs="+", choices=ENCODERS,
                        default=["clip_l14", "dinov2_b14", "convnext_tiny"])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

    paths, labels, _ = collect_images(args.data_dir)
    values = split_data(paths, labels)
    splits = dict(zip(("train", "validation", "test"), values))
    y = {name: np.asarray(split[1]) for name, split in splits.items()}
    target_device = device()
    print(f"device={target_device}; " + ", ".join(f"{k}={len(v[0])}" for k, v in splits.items()))

    all_features = {
        name: embeddings(name, ENCODERS[name], splits, args.output_dir,
                         target_device, args.batch_size)
        for name in args.encoders
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    selected, validation_probabilities, individual = {}, {}, {}
    for name, features in all_features.items():
        best, candidates = choose_classifier(features["train"], y["train"], cv)
        mean, std, c = best
        fitted = classifier(c).fit(features["train"], y["train"])
        probabilities = fitted.predict_proba(features["validation"])
        validation_probabilities[name] = probabilities
        validation_accuracy = accuracy_score(y["validation"], probabilities.argmax(axis=1))
        selected[name] = (c, fitted)
        individual[name] = {
            "selected_C": c, "train_cv_mean": mean, "train_cv_std": std,
            "validation_accuracy": validation_accuracy,
            "cv_candidates": [{"C": row[2], "mean": row[0], "std": row[1]}
                              for row in candidates],
        }
        print(f"{name}: C={c:g}, train CV={mean:.4f}±{std:.4f}, val={validation_accuracy:.4f}")

    best_ensemble = None
    for weights in ensemble_candidates(args.encoders):
        probabilities = weighted_probabilities(validation_probabilities, weights)
        score = accuracy_score(y["validation"], probabilities.argmax(axis=1))
        complexity = len(weights)
        candidate = (score, -complexity, weights)
        if best_ensemble is None or candidate[:2] > best_ensemble[:2]:
            best_ensemble = candidate
    validation_accuracy, _, weights = best_ensemble
    print(f"selected ensemble: {weights}; val={validation_accuracy:.4f}")

    train_val_labels = np.concatenate((y["train"], y["validation"]))
    test_probabilities, final_models = {}, {}
    for name, weight in weights.items():
        features = all_features[name]
        train_val = np.concatenate((features["train"], features["validation"]))
        c = selected[name][0]
        fitted = classifier(c).fit(train_val, train_val_labels)
        final_models[name] = fitted
        test_probabilities[name] = fitted.predict_proba(features["test"])
    test_probability = weighted_probabilities(test_probabilities, weights)
    predictions = test_probability.argmax(axis=1)
    metrics = {
        "selection_protocol": "C selected by train-only 5-fold CV; ensemble selected on validation; test used once after selection",
        "split_sizes": {name: len(split[0]) for name, split in splits.items()},
        "individual_models": individual,
        "ensemble_weights": weights,
        "validation_accuracy": validation_accuracy,
        "test_accuracy": accuracy_score(y["test"], predictions),
        "confusion_matrix": confusion_matrix(y["test"], predictions).tolist(),
        "classification_report": classification_report(
            y["test"], predictions, target_names=CLASS_NAMES, output_dict=True, zero_division=0
        ),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    joblib.dump({"models": final_models, "weights": weights, "encoders": ENCODERS,
                 "class_names": CLASS_NAMES, "variant": "flip_tta_average"},
                args.output_dir / "best_ensemble.joblib")
    print(json.dumps({key: metrics[key] for key in (
        "ensemble_weights", "validation_accuracy", "test_accuracy",
        "confusion_matrix", "classification_report")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
