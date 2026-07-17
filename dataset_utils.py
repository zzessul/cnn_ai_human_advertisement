import hashlib
import unicodedata
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split


SEED = 42
CLASS_NAMES = ["인간광고", "AI광고"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def candidate_class_dirs(root: Path) -> list[tuple[Path, int]]:
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
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_images(root: Path) -> tuple[list[str], list[int], dict]:
    records, skipped_invalid = [], []
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

    by_hash, conflicts = {}, []
    for path, label, digest in records:
        if digest in by_hash and by_hash[digest][1] != label:
            conflicts.append([str(by_hash[digest][0]), str(path)])
            continue
        by_hash.setdefault(digest, (path, label))

    paths = [str(value[0]) for value in by_hash.values()]
    labels = [value[1] for value in by_hash.values()]
    audit = {
        "source_root": str(root.resolve()),
        "files_before_deduplication": len(records),
        "exact_duplicates_removed": len(records) - len(by_hash),
        "invalid_images_skipped": skipped_invalid,
        "cross_label_hash_conflicts": conflicts,
        "class_counts_after_deduplication": {
            CLASS_NAMES[index]: labels.count(index) for index in range(2)
        },
    }
    return paths, labels, audit


def split_data(paths, labels):
    categories = []
    for path in paths:
        name = Path(path).name
        category = next((value for value in ("model", "product", "character", "legacy")
                         if f"_{value}_" in name), "legacy")
        categories.append(category)
    strata = [f"{label}:{category}" for label, category in zip(labels, categories)]
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=0.30, random_state=SEED, stratify=strata
    )
    category_by_path = dict(zip(paths, categories))
    temp_strata = [f"{label}:{category_by_path[path]}"
                   for path, label in zip(temp_paths, temp_labels)]
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, random_state=SEED,
        stratify=temp_strata
    )
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)
