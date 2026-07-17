import argparse
import hashlib
import json
import shutil
import unicodedata
from pathlib import Path

from PIL import Image, UnidentifiedImageError


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = ["인간광고", "AI광고"]


def normalized_name(path: Path) -> str:
    return unicodedata.normalize("NFC", path.name).replace(" ", "")


def infer_category(path: Path) -> str:
    names = [normalized_name(parent) for parent in path.parents]
    if any("모델(사람)광고" in name for name in names):
        return "model"
    if any("제품광고" in name for name in names):
        return "product"
    if any("캐릭터광고" in name for name in names):
        return "character"
    return "legacy"


def class_directories(root: Path):
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        name = normalized_name(path)
        if name in {"AI", "AI광고"}:
            yield path, 1
        elif name in {"인간", "인간광고", "캐릭터"}:
            yield path, 0


def nearest_label(path: Path, root: Path):
    """파일에서 가장 가까운(최하위) 라벨 폴더를 우선한다."""
    for parent in path.parents:
        if parent == root.parent:
            break
        name = normalized_name(parent)
        if name in {"AI", "AI광고"}:
            return 1
        if name in {"인간", "인간광고", "캐릭터"}:
            return 0
        if parent == root:
            break
    return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(path: Path) -> int:
    with Image.open(path) as image:
        image = image.convert("L").resize((9, 8))
        pixels = list(image.getdata())
    value = 0
    for row in range(8):
        for column in range(8):
            value = (value << 1) | (
                pixels[row * 9 + column] > pixels[row * 9 + column + 1]
            )
    return value


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def collect(root: Path, source: str):
    records, invalid, excluded = [], [], []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        label = nearest_label(path, root)
        if label is None:
            excluded.append(str(path))
            continue
        if "contact_sheet" in path.name.lower():
            excluded.append(str(path))
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            records.append({
                "path": path,
                "label": label,
                "source": source,
                "category": infer_category(path),
                "sha256": sha256(path),
                "dhash": dhash(path),
            })
        except (OSError, UnidentifiedImageError):
            invalid.append(str(path))
    return records, invalid, excluded


def deduplicate(records, existing=None, near_threshold=2):
    kept = [] if existing is None else list(existing)
    output, duplicates = [], []
    hashes = {record["sha256"]: record for record in kept}
    by_label = {
        label: [record for record in kept if record["label"] == label]
        for label in range(2)
    }
    for record in records:
        match, reason = hashes.get(record["sha256"]), "exact_sha256"
        if match is None:
            for candidate in by_label[record["label"]]:
                if hamming(record["dhash"], candidate["dhash"]) <= near_threshold:
                    match, reason = candidate, "near_duplicate_dhash"
                    break
        if match is not None:
            duplicates.append({
                "excluded": record["path"].name,
                "matched": match["path"].name,
                "reason": reason,
            })
            continue
        output.append(record)
        kept.append(record)
        hashes[record["sha256"]] = record
        by_label[record["label"]].append(record)
    return output, duplicates


def copy_records(records, output: Path):
    for folder in CLASS_NAMES:
        (output / folder).mkdir(parents=True, exist_ok=True)
    manifest = []
    counters = {}
    for record in records:
        key = record["source"], record["category"], record["label"]
        counters[key] = counters.get(key, 0) + 1
        suffix = record["path"].suffix.lower()
        filename = f"{record['source']}_{record['category']}_{counters[key]:04d}{suffix}"
        destination = output / CLASS_NAMES[record["label"]] / filename
        shutil.copy2(record["path"], destination)
        manifest.append({
            "file": str(destination.relative_to(output)),
            "label": CLASS_NAMES[record["label"]],
            "source": record["source"],
            "category": record["category"],
            "sha256": record["sha256"],
        })
    return manifest


def counts(records):
    return {CLASS_NAMES[label]: sum(r["label"] == label for r in records) for label in range(2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-dir", type=Path, required=True)
    parser.add_argument("--b-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--near-threshold", type=int, default=2)
    args = parser.parse_args()

    a_records, a_invalid, a_excluded = collect(args.a_dir, "A_minus_B")
    b_records, b_invalid, b_excluded = collect(args.b_dir, "B_original")
    b_unique, b_duplicates = deduplicate(b_records, near_threshold=args.near_threshold)
    a_unique, a_duplicates = deduplicate(
        a_records, existing=b_unique, near_threshold=args.near_threshold
    )

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    combined = b_unique + a_unique
    manifest = copy_records(combined, args.output_dir)
    report = {
        "A_scanned": counts(a_records),
        "B_scanned": counts(b_records),
        "B_unique": counts(b_unique),
        "A_minus_B_unique": counts(a_unique),
        "combined_unique": counts(combined),
        "near_duplicate_hamming_threshold": args.near_threshold,
        "B_duplicates_removed": b_duplicates,
        "A_duplicates_found_in_B_or_A": a_duplicates,
        "invalid_images": b_invalid + a_invalid,
        "non_training_images_excluded": b_excluded + a_excluded,
    }
    (args.output_dir / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in (
        "A_scanned", "B_scanned", "B_unique", "A_minus_B_unique", "combined_unique"
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
