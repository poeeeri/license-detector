from __future__ import annotations

import logging

from ocr_labeler.cli import parse_args
from ocr_labeler.env import load_dotenv_if_present


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    load_dotenv_if_present()
    config = parse_args()

    from ocr_labeler.pipeline import run

    run(config)


if __name__ == "__main__":
    main()