import argparse
import cv2
import logging
import torch
from pathlib import Path
from model_impl import My_LicensePlate_Model

def setup_logging():
    Path("./data").mkdir(exist_ok=True)
    logger = logging.getLogger("LicensePlateDetector")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler("./data/log_file.log", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def annotate_frame(model: My_LicensePlate_Model, frame):
    annotated = frame.copy()
    for plate in model.detect_plates(frame):
        x1, y1, x2, y2 = plate["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # уверенность модели или оср текст
        label = plate.get("text") or f"{plate['confidence']:.2f}"
        cv2.putText( annotated, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    return annotated


def process_video(model, input_path, output_path, logger, show=False):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Не смог открыть видео: {input_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    frame_count = 0
    logger.info("Старт обработки видео: input=%s output=%s", input_path, output_path)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.info("трансляция видео завершилась после %s фреймов", frame_count)
                break
            annotated = annotate_frame(model, frame)
            out.write(annotated)
            frame_count += 1

            if frame_count % 100 == 0:
                logger.info("Processed %s frames", frame_count)

            if show:
                cv2.imshow('Detection', annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Обработка видео прервана пользователем на фрейме %s", frame_count)
                    break
    finally:
        cap.release()
        out.release()
        if show:
            cv2.destroyAllWindows()

    logger.info("Video saved to %s", output_path)


def process_camera(model: My_LicensePlate_Model, cam_id: int, logger: logging.Logger):
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise ValueError(f"Не смог открыть камеру: {cam_id}")

    logger.info("старт обработки камеры: cam_id=%s", cam_id)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.warning("Не удалось получить кадр с камеры с идентификатором cam_id=%s", cam_id)
                break

            annotated = annotate_frame(model, frame)
            cv2.imshow('License Plate Detection', annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Обработка с камеры прервана пользователем")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("обработка c камеры завершена: cam_id=%s", cam_id)


def main():
    logger = setup_logging()

    if torch.cuda.is_available():
        default_device = "cuda"
        print("CUDA доступна")
    else:
        default_device = "cpu"
        print("CUDA не обнаружена")

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["video", "cam"], required=True)
    parser.add_argument("--input", help="Path to video file (for video mode)")
    parser.add_argument("--output", default="output.mp4", help="Output video file")
    parser.add_argument("--cam_id", type=int, default=0)
    parser.add_argument("--weights", default="weights/best.pt")
    parser.add_argument('--device', default=default_device, choices=['cuda', 'cpu'], 
                        help='Device: cuda (GPU) or cpu (default: auto)')
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--ocr-backend", default="hf", choices=["hf", "huggingface", "trocr"])
    parser.add_argument("--ocr-model", default="best_model_export")
    parser.add_argument("--ocr-device", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--ocr-max-length", type=int)
    parser.add_argument("--ocr-num-beams", type=int)
    parser.add_argument("--ocr-padding", type=float, default=0.05)
    parser.add_argument("--disable-ocr", action="store_true")
    args = parser.parse_args()

    logger.info(
        "Приложение запущено: mode=%s input=%s output=%s cam_id=%s weights=%s device=%s conf=%s, ocr_model=%s",
        args.mode,
        args.input,
        args.output,
        args.cam_id,
        args.weights,
        args.device,
        args.conf,
        None if args.disable_ocr else args.ocr_model
    )

    model = My_LicensePlate_Model(
        args.weights,
        device=args.device,
        conf_threshold=args.conf,
        logger=logger,
        ocr_backend=args.ocr_backend,
        ocr_model_path=None if args.disable_ocr else args.ocr_model,
        ocr_device=args.ocr_device,
        ocr_max_length=args.ocr_max_length,
        ocr_num_beams=args.ocr_num_beams,
        crop_padding=args.ocr_padding
    )

    if args.mode == "video":
        if not args.input:
            logger.error("--input required for video mode")
            return
        process_video(model, args.input, args.output, logger=logger, show=True)
    else:
        process_camera(model, args.cam_id, logger=logger)

    logger.info("Приложение успешно завершилось")


if __name__ == "__main__":
    main()