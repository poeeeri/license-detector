from __future__ import annotations
from typing import Protocol
import numpy as np


# это не настоящая модель а общий интерфейс ocr, обертка, чтобы если что 
# заменить модель (сейчас подается тестовая модель TrOCR, дообученная на 5 эпохах без экспериментов над ней (baseline))
class OCRBackend(Protocol):
    def predict(self, image: np.ndarray) -> str:
        ...

    def predict_batch(self, images: list[np.ndarray]) -> list[str]:
        ...