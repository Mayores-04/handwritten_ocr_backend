"""Dataset discovery helpers for the handwriting OCR training pipeline.

The project currently keeps the real IAM word images under:

    data/full_dataset/iam_words/words/<group>/<form>/<word-id>.png

and the IAM metadata under:

    data/full_dataset/words_new.txt

This module keeps that real layout usable without copying 100k+ image files
into flat train/val folders.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class DatasetSample:
    """A single OCR training sample."""

    image_path: Path
    text: str
    source_id: str
    source: str


@dataclass
class DatasetDiscoveryResult:
    """Summary and samples found in the local dataset."""

    dataset_path: Path
    source: str = "none"
    labels_path: Path | None = None
    images_root: Path | None = None
    train_samples: list[DatasetSample] = field(default_factory=list)
    val_samples: list[DatasetSample] = field(default_factory=list)
    labels_count: int = 0
    matched_count: int = 0
    missing_count: int = 0
    skipped_count: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def total_samples(self) -> int:
        return len(self.train_samples) + len(self.val_samples)

    def to_dict(self, preview_count: int = 5) -> dict:
        preview = self.train_samples[:preview_count] + self.val_samples[:preview_count]
        return {
            "dataset_path": str(self.dataset_path),
            "source": self.source,
            "labels_path": str(self.labels_path) if self.labels_path else None,
            "images_root": str(self.images_root) if self.images_root else None,
            "labels_count": self.labels_count,
            "matched_count": self.matched_count,
            "missing_count": self.missing_count,
            "skipped_count": self.skipped_count,
            "train_count": len(self.train_samples),
            "val_count": len(self.val_samples),
            "total_samples": self.total_samples,
            "warnings": self.warnings,
            "preview": [
                {
                    "image": str(sample.image_path),
                    "text": sample.text,
                    "source_id": sample.source_id,
                    "source": sample.source,
                }
                for sample in preview[:preview_count]
            ],
        }


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _read_simple_labels(labels_file: Path) -> tuple[dict[str, str], int]:
    """Read labels.txt using the project format: filename,transcription."""

    labels: dict[str, str] = {}
    skipped = 0

    with labels_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue

            if "," not in raw:
                skipped += 1
                continue

            filename, text = raw.split(",", 1)
            filename = filename.strip()
            text = text.strip()

            if not filename or not text or text == "[TRANSCRIPTION NEEDED]":
                skipped += 1
                continue

            labels[filename] = text
            labels[Path(filename).stem] = text

    return labels, skipped


def _discover_flat_samples(dataset_path: Path) -> DatasetDiscoveryResult:
    """Discover samples in data/train + data/val when those folders are populated."""

    result = DatasetDiscoveryResult(dataset_path=dataset_path)
    labels_file = dataset_path / "labels.txt"
    if not labels_file.exists():
        result.warnings.append(f"No labels.txt found at {labels_file}")
        return result

    labels, skipped = _read_simple_labels(labels_file)
    result.labels_path = labels_file
    result.labels_count = len({key for key in labels if "." in key})
    result.skipped_count += skipped

    for split, bucket in (("train", result.train_samples), ("val", result.val_samples)):
        split_dir = dataset_path / split
        if not split_dir.exists():
            result.warnings.append(f"Split directory not found: {split_dir}")
            continue

        for image_path in sorted(split_dir.rglob("*")):
            if not image_path.is_file() or not _is_image(image_path):
                continue

            text = labels.get(image_path.name) or labels.get(image_path.stem)
            if not text:
                result.missing_count += 1
                continue

            bucket.append(
                DatasetSample(
                    image_path=image_path,
                    text=text,
                    source_id=image_path.name,
                    source="flat",
                )
            )

    result.matched_count = result.total_samples
    if result.total_samples:
        result.source = "flat_train_val"
        result.images_root = dataset_path

    return result


def _iam_image_path(images_root: Path, word_id: str) -> Path:
    parts = word_id.split("-")
    if len(parts) < 2:
        return images_root / f"{word_id}.png"

    group = parts[0]
    form = "-".join(parts[:2])
    return images_root / group / form / f"{word_id}.png"


def _parse_iam_words(
    words_file: Path,
    images_root: Path,
    stop_after: int | None = None,
) -> tuple[list[DatasetSample], int, int, int]:
    """Parse IAM word metadata and return only samples whose image file exists."""

    samples: list[DatasetSample] = []
    labels_count = 0
    missing_count = 0
    skipped_count = 0

    with words_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue

            parts = raw.split()
            if len(parts) < 9:
                skipped_count += 1
                continue

            labels_count += 1
            word_id = parts[0]
            segmentation_status = parts[1].lower()
            text = parts[-1].replace("|", " ").strip()

            if segmentation_status != "ok" or not text:
                skipped_count += 1
                continue

            image_path = _iam_image_path(images_root, word_id)
            if not image_path.exists():
                missing_count += 1
                continue

            samples.append(
                DatasetSample(
                    image_path=image_path,
                    text=text,
                    source_id=word_id,
                    source="iam_words",
                )
            )
            if stop_after is not None and len(samples) >= stop_after:
                break

    return samples, labels_count, missing_count, skipped_count


def _split_samples(
    samples: Iterable[DatasetSample],
    test_size: float,
    seed: int,
    max_samples: int | None,
) -> tuple[list[DatasetSample], list[DatasetSample]]:
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)

    if max_samples is not None and max_samples > 0:
        shuffled = shuffled[:max_samples]

    if not shuffled:
        return [], []

    test_size = min(max(test_size, 0.01), 0.9)
    split_idx = int(len(shuffled) * (1 - test_size))
    split_idx = min(max(split_idx, 1), len(shuffled) - 1) if len(shuffled) > 1 else len(shuffled)

    return shuffled[:split_idx], shuffled[split_idx:]


def _discover_iam_word_samples(
    dataset_path: Path,
    test_size: float,
    seed: int,
    max_samples: int | None,
) -> DatasetDiscoveryResult:
    result = DatasetDiscoveryResult(dataset_path=dataset_path)
    words_file = dataset_path / "full_dataset" / "words_new.txt"
    images_root = dataset_path / "full_dataset" / "iam_words" / "words"

    if not words_file.exists():
        result.warnings.append(f"IAM metadata not found: {words_file}")
        return result

    if not images_root.exists():
        result.warnings.append(f"IAM image root not found: {images_root}")
        return result

    samples, labels_count, missing_count, skipped_count = _parse_iam_words(
        words_file,
        images_root,
        stop_after=max_samples,
    )
    train_samples, val_samples = _split_samples(samples, test_size, seed, max_samples)

    result.source = "iam_words"
    result.labels_path = words_file
    result.images_root = images_root
    result.train_samples = train_samples
    result.val_samples = val_samples
    result.labels_count = labels_count
    result.matched_count = len(samples)
    result.missing_count = missing_count
    result.skipped_count = skipped_count

    if not samples:
        result.warnings.append("IAM metadata was found, but no matching image files were found.")

    return result


def discover_labeled_image_samples(
    dataset_path: str | Path,
    test_size: float = 0.2,
    seed: int = 1337,
    max_samples: int | None = None,
    prefer_iam: bool = True,
) -> DatasetDiscoveryResult:
    """Discover real labeled OCR samples from the project dataset folder."""

    dataset = Path(dataset_path).resolve()
    if not dataset.exists():
        result = DatasetDiscoveryResult(dataset_path=dataset)
        result.warnings.append(f"Dataset path does not exist: {dataset}")
        return result

    flat = _discover_flat_samples(dataset)
    iam = _discover_iam_word_samples(dataset, test_size, seed, max_samples)

    if prefer_iam and iam.total_samples:
        if flat.total_samples:
            iam.warnings.append(
                f"Using IAM words dataset; flat train/val also has {flat.total_samples} samples."
            )
        return iam

    if flat.total_samples:
        return flat

    if iam.total_samples:
        return iam

    combined = DatasetDiscoveryResult(dataset_path=dataset)
    combined.labels_path = flat.labels_path or iam.labels_path
    combined.images_root = iam.images_root or flat.images_root
    combined.labels_count = flat.labels_count + iam.labels_count
    combined.missing_count = flat.missing_count + iam.missing_count
    combined.skipped_count = flat.skipped_count + iam.skipped_count
    combined.warnings = flat.warnings + iam.warnings
    combined.warnings.append("No usable labeled image samples were discovered.")
    return combined


def get_dataset_summary(dataset_path: str | Path = "data") -> dict:
    """Return a JSON-safe dataset summary for health checks and diagnostics."""

    result = discover_labeled_image_samples(dataset_path=dataset_path, max_samples=20)
    return result.to_dict(preview_count=3)
