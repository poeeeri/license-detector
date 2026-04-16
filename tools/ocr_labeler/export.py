from __future__ import annotations
import csv
from pathlib import Path
from typing import Any
from .config import LABEL_FIELDNAMES
from .labels import normalize_plate_text


def load_existing_rows(labels_path: Path) -> dict[str, dict[str, Any]]:
    if not labels_path.exists():
        return {}
    with labels_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    return {str(Path(row["crop_path"]).resolve()): row for row in rows if row.get("crop_path")}


def save_rows(rows: list[dict[str, Any]], labels_path: Path) -> None:
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with labels_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=LABEL_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_paddleocr_labels(
    rows: list[dict[str, Any]],
    output_dir: Path,
    paddleocr_dir: Path,
):
    paddleocr_dir.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("needs_review", "")).lower() == "true":
            continue
        text = normalize_plate_text(row.get("text_normalized", ""))
        crop_path = row.get("crop_path", "")
        split = row.get("split", "")
        if not text or not crop_path or not split:
            continue
        by_split.setdefault(split, []).append(row)

    for split, split_rows in by_split.items():
        label_path = paddleocr_dir / f"rec_gt_{split}.txt"
        with label_path.open("w", encoding="utf-8", newline="\n") as file:
            for row in split_rows:
                crop_path = Path(str(row["crop_path"]))
                crop_rel = crop_path.resolve().relative_to(output_dir.resolve()).as_posix()
                text = normalize_plate_text(row["text_normalized"])
                file.write(f"{crop_rel}\t{text}\n")