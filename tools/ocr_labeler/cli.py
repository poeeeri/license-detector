from __future__ import annotations
import argparse
import os
from pathlib import Path
from .config import LabelerConfig


def parse_args() -> LabelerConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--provider",
        choices=["openai", "openrouter"],
        default=os.getenv("OCR_LABELER_PROVIDER", "openrouter"),
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--padding", type=float, default=0.2)
    parser.add_argument("--min-confidence", type=float, default=0.85)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--detail", choices=["low", "high", "auto"], default="high")
    args = parser.parse_args()

    return LabelerConfig(
        data_root=Path(args.data_root).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        provider=args.provider,
        model=args.model,
        splits=args.splits,
        limit=args.limit,
        padding=args.padding,
        min_confidence=args.min_confidence,
        resume=args.resume,
        sleep_seconds=args.sleep_seconds,
        detail=args.detail,
    )