from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Any
from .config import LabelerConfig
from .export import load_existing_rows, save_rows, write_paddleocr_labels
from .labels import build_error_row, build_success_row
from .preprocess import crop_plate, iter_plate_boxes
from .providers import build_client, read_plate_text


def run(config: LabelerConfig):
    client, model = build_client(config.provider, config.model)
    prepare_output_dirs(config)

    existing_rows = load_existing_rows(config.labels_path) if config.resume else {}
    rows: list[dict[str, Any]] = list(existing_rows.values())
    processed = 0

    logging.info("using provider: %s", config.provider)
    logging.info("using model: %s", model)
    logging.info("reading YOLO dataset from: %s", config.data_root)

    for item in iter_plate_boxes(config.data_root, config.splits):
        crop_rel = Path(item.split) / f"{item.image_path.stem}_{item.plate_index}.jpg"
        crop_path = config.crops_dir / crop_rel
        crop_key = str(crop_path.resolve())

        if crop_key in existing_rows:
            logging.info("skipping existing crop %s", crop_rel)
            continue

        if config.limit > 0 and processed >= config.limit:
            break

        crop_box = crop_plate(
            image_path=item.image_path,
            bbox=item.bbox_xyxy,
            output_path=crop_path,
            padding=config.padding,
        )

        logging.info("labeling %s", crop_rel)
        try:
            payload = read_plate_text(
                client=client,
                provider=config.provider,
                model=model,
                crop_path=crop_path,
                detail=config.detail,
            )
            row = build_success_row(
                item=item,
                crop_path=crop_path,
                crop_box=crop_box,
                payload=payload,
                min_confidence=config.min_confidence,
            )
        except Exception as exc:
            logging.exception("failed to label %s", crop_rel)
            row = build_error_row(item=item, crop_path=crop_path, crop_box=crop_box, error=exc)

        rows.append(row)
        save_rows(rows, config.labels_path)
        write_paddleocr_labels(rows, config.output_dir, config.paddleocr_dir)
        processed += 1

        if config.sleep_seconds > 0:
            time.sleep(config.sleep_seconds)

    save_rows(rows, config.labels_path)
    write_paddleocr_labels(rows, config.output_dir, config.paddleocr_dir)
    logging.info("finished. saved %s rows to %s", len(rows), config.labels_path)


def prepare_output_dirs(config: LabelerConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.crops_dir.mkdir(parents=True, exist_ok=True)
    config.paddleocr_dir.mkdir(parents=True, exist_ok=True)