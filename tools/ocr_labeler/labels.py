from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any
from .config import NORMALIZED_PLATE_PATTERN
from .preprocess import PlateBox


def build_success_row(
    item: PlateBox,
    crop_path: Path,
    crop_box: tuple[int, int, int, int],
    payload: dict[str, Any],
    min_confidence: float,
) -> dict[str, Any]:
    text_raw = normalize_text(payload.get("text_raw", ""))
    text_normalized = normalize_plate_text(payload.get("text_normalized", text_raw))
    confidence = float(payload.get("confidence", 0) or 0)
    return {
        "split": item.split,
        "source_image": str(item.image_path.resolve()),
        "source_label": str(item.label_path.resolve()),
        "crop_path": str(crop_path.resolve()),
        "plate_index": item.plate_index,
        "class_id": item.class_id,
        "source_bbox_xyxy": json.dumps(item.bbox_xyxy),
        "crop_box_xyxy": json.dumps(crop_box),
        "text_raw": text_raw,
        "text_normalized": text_normalized,
        "confidence": confidence,
        "needs_review": confidence < min_confidence or not text_normalized,
        "unreadable_reason": normalize_text(payload.get("unreadable_reason", "")),
        "raw_output": payload.get("raw_output", ""),
    }


def build_error_row(
    item: PlateBox,
    crop_path: Path,
    crop_box: tuple[int, int, int, int],
    error: Exception,
) -> dict[str, Any]:
    return {
        "split": item.split,
        "source_image": str(item.image_path.resolve()),
        "source_label": str(item.label_path.resolve()),
        "crop_path": str(crop_path.resolve()),
        "plate_index": item.plate_index,
        "class_id": item.class_id,
        "source_bbox_xyxy": json.dumps(item.bbox_xyxy),
        "crop_box_xyxy": json.dumps(crop_box),
        "text_raw": "",
        "text_normalized": "",
        "confidence": 0.0,
        "needs_review": True,
        "unreadable_reason": str(error),
        "raw_output": "",
    }


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_plate_text(value: object) -> str:
    text = normalize_text(value).upper()
    return "".join(re.findall(NORMALIZED_PLATE_PATTERN, text))