from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from .config import SUPPORTED_IMAGE_EXTENSIONS


@dataclass(frozen=True)
class PlateBox:
    split: str
    image_path: Path
    label_path: Path
    plate_index: int
    class_id: int
    bbox_xyxy: tuple[int, int, int, int]


def iter_plate_boxes(data_root: Path, splits: list[str]) -> list[PlateBox]:
    items: list[PlateBox] = []
    for split in splits:
        images_dir = data_root / split / "images"
        labels_dir = data_root / split / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            logging.warning("skipping split without images/labels: %s", split)
            continue

        for label_path in sorted(labels_dir.glob("*.txt")):
            image_path = find_image(images_dir, label_path.stem)
            if image_path is None:
                logging.warning("image not found for label %s", label_path)
                continue

            with Image.open(image_path) as image:
                image_width, image_height = image.size

            for plate_index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
                parsed = parse_yolo_line(line, image_width, image_height)
                if parsed is None:
                    continue
                class_id, bbox_xyxy = parsed
                items.append(
                    PlateBox(
                        split=split,
                        image_path=image_path,
                        label_path=label_path,
                        plate_index=plate_index,
                        class_id=class_id,
                        bbox_xyxy=bbox_xyxy,
                    )
                )
    return items


def find_image(images_dir: Path, stem: str) -> Path | None:
    for extension in SUPPORTED_IMAGE_EXTENSIONS:
        path = images_dir / f"{stem}{extension}"
        if path.exists():
            return path
    return None


def parse_yolo_line(
    line: str,
    image_width: int,
    image_height: int,
) -> tuple[int, tuple[int, int, int, int]] | None:
    parts = line.split()
    if len(parts) < 5:
        return None

    class_id = int(float(parts[0]))
    x_center, y_center, width, height = [float(value) for value in parts[1:5]]

    x1 = int((x_center - width / 2) * image_width)
    y1 = int((y_center - height / 2) * image_height)
    x2 = int((x_center + width / 2) * image_width)
    y2 = int((y_center + height / 2) * image_height)
    return class_id, clamp_box((x1, y1, x2, y2), image_width, image_height)


def clamp_box(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(image_width, x1)),
        max(0, min(image_height, y1)),
        max(0, min(image_width, x2)),
        max(0, min(image_height, y2)),
    )


def crop_plate(
    image_path: Path,
    bbox: tuple[int, int, int, int],
    output_path: Path,
    padding: float,
) -> tuple[int, int, int, int]:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_width, image_height = image.size
        x1, y1, x2, y2 = bbox
        box_width = max(1, x2 - x1)
        box_height = max(1, y2 - y1)
        pad_x = int(box_width * padding)
        pad_y = int(box_height * padding)
        crop_box = clamp_box(
            (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y),
            image_width,
            image_height,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.crop(crop_box).save(output_path, format="JPEG", quality=95)
        return crop_box