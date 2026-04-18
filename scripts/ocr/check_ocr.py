import cv2
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from ocr.factory import build_ocr
from ocr.postprocess import normalize_plate_text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True)
    parser.add_argument("--label")
    parser.add_argument("--model", default="best_model_export")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.img)
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"невозможн прочиать изображение: {image_path}")

    ocr = build_ocr("hf", args.model, device=args.device)
    raw_text = ocr.predict(img)
    prediction = normalize_plate_text(raw_text)

    if args.label:
        ground_truth = normalize_plate_text(args.label)
        print("ground_truth:", ground_truth)

    print("prediction:", prediction)
    print("raw_prediction:", raw_text)

    if args.label:
        print("exact_match:", prediction == ground_truth)


if __name__ == "__main__":
    main()