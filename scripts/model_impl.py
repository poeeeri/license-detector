from ultralytics import YOLO
from pathlib import Path
import logging
from typing import List, Dict, Any
import numpy as np

class My_LicensePlate_Model:
    def __init__(self, weights_path: str, device: str = 'cpu', conf_threshold: float = 0.30, logger=None):
        self.weights_path = Path(weights_path)
        self.device = device
        self.conf_threshold = conf_threshold
        self.logger = logger or self._get_default_logger()
        
        self.logger.info(f"Загрузка модели из {self.weights_path}, device={self.device}")
        
        self.model = YOLO(str(self.weights_path))
        self.model.to(self.device)
        
        self.logger.info("Модель загружена!")
    
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
        
        self.logger.debug(f"Найдено {len(detections)} номеров на кадре")
        return detections