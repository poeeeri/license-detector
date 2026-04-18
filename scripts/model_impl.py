from ultralytics import YOLO
from pathlib import Path
import logging
from typing import List, Dict, Any
import numpy as np
from ocr import OCRBackend, build_ocr
from ocr.postprocess import normalize_plate_text


class My_LicensePlate_Model:
    def __init__(
        self,
        weights_path: str,
        device: str = 'cpu',
        conf_threshold: float = 0.30,
        logger=None,
        ocr_backend: str = 'hf',
        ocr_model_path: str | None = None,
        ocr_device: str | None = None,
        ocr_max_length: int | None = None,
        ocr_num_beams: int | None = None,
        crop_padding: float = 0.05,
    ):
        self.weights_path = Path(weights_path)
        self.device = device
        self.conf_threshold = conf_threshold
        self.logger = logger or self._get_default_logger()
        self.crop_padding = crop_padding
        self.ocr: OCRBackend | None = None
        
        self.logger.info(f"Загрузка модели из {self.weights_path}, device={self.device}")
        
        self.model = YOLO(str(self.weights_path))
        self.model.to(self.device)
        
        self.logger.info("Модель загружена!")
    
        if ocr_model_path:
            self.logger.info(
                "загрузка OCR backend=%s model=%s device=%s",
                ocr_backend,
                ocr_model_path,
                ocr_device or self.device,
            )
            self.ocr = build_ocr(
                backend=ocr_backend,
                model_path=ocr_model_path,
                device=ocr_device or self.device,
                max_length=ocr_max_length,
                num_beams=ocr_num_beams,
            )
            self.logger.info("OCR загружена!")
    
    @staticmethod
    def _get_default_logger() -> logging.Logger:
        logger = logging.getLogger("LicensePlateDetector")
        logger.setLevel(logging.INFO)
        log_dir = Path("./data")
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "log_file.log", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.propagate = False
        return logger
    
    def detect_plates(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if frame is None or frame.size == 0:
            self.logger.warning("Received empty frame")
            return []
        
        results = self.model(frame, device=self.device, conf=self.conf_threshold, verbose=False)
        
        detections = []
        crops = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = map(int, box)
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf)
                })

                if self.ocr is not None:
                    crops.append(self._crop_plate(frame, [x1, y1, x2, y2]))

        if self.ocr is not None and crops:
            try:
                raw_texts = self.ocr.predict_batch(crops)
            except Exception:
                self.logger.exception("OCR batch failed")
                raw_texts = [""] * len(crops)

            for detection, raw_text in zip(detections, raw_texts):
                detection["text_raw"] = raw_text
                detection["text"] = normalize_plate_text(raw_text)

        self.logger.debug(f"Найдено {len(detections)} номеров на кадре")
        return detections

# добавляем функцию кропа изображения для ocr
    def _crop_plate(self, frame: np.ndarray, bbox: list[int]) -> np.ndarray:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        pad_x = int((x2 - x1) * self.crop_padding)
        pad_y = int((y2 - y1) * self.crop_padding)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(width, x2 + pad_x)
        y2 = min(height, y2 + pad_y)

        return frame[y1:y2, x1:x2]