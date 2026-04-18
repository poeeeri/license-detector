from __future__ import annotations
from pathlib import Path
from .base import OCRBackend
from .hf_ocr import HuggingFaceOCR


def build_ocr(
    backend: str,
    model_path: str | Path,
    device: str = "cpu",
    max_length: int | None = None,
    num_beams: int | None = None,
) -> OCRBackend:
    normalized_backend = backend.lower()
    if normalized_backend in {"hf", "huggingface", "trocr"}:
        return HuggingFaceOCR(
            model_path=model_path,
            device=device,
            max_length=max_length,
            num_beams=num_beams,
        )

    raise ValueError(f"неподдерживаемый OCR backend: {backend} (пока есть только huggingface)")