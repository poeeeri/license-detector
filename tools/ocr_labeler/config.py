from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
DEFAULT_OPENAI_MODEL = "gpt-5-mini"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-5-mini"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
CYRILLIC_UPPER_CHARS = "\u0410-\u042f\u0401"
NORMALIZED_PLATE_PATTERN = rf"[A-Z0-9{CYRILLIC_UPPER_CHARS}]"

LABEL_FIELDNAMES = [
    "split",
    "source_image",
    "source_label",
    "crop_path",
    "plate_index",
    "class_id",
    "source_bbox_xyxy",
    "crop_box_xyxy",
    "text_raw",
    "text_normalized",
    "confidence",
    "needs_review",
    "unreadable_reason",
    "raw_output",
]

PLATE_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "text_raw": {"type": "string"},
        "text_normalized": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "unreadable_reason": {"type": "string"},
    },
    "required": ["text_raw", "text_normalized", "confidence", "unreadable_reason"],
}


@dataclass(frozen=True)
class LabelerConfig:
    data_root: Path
    output_dir: Path
    provider: str
    model: str | None
    splits: list[str]
    limit: int
    padding: float
    min_confidence: float
    resume: bool
    sleep_seconds: float
    detail: str

    @property
    def crops_dir(self) -> Path:
        return self.output_dir / "crops"

    @property
    def labels_path(self) -> Path:
        return self.output_dir / "labels.csv"

    @property
    def paddleocr_dir(self) -> Path:
        return self.output_dir / "paddleocr"