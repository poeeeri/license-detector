from __future__ import annotations
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import torch
from transformers import AutoProcessor, VisionEncoderDecoderModel


class HuggingFaceOCR:
    def __init__(self, model_path, device ='cpu', max_length = None, num_beams= None):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"OCR модели по этому пути не сущ-т: {self.model_path}")

        self.processor = AutoProcessor.from_pretrained(str(self.model_path), local_files_only=True)
        self.model = VisionEncoderDecoderModel.from_pretrained(
            str(self.model_path),
            local_files_only=True,
        )
        self.device = self._resolve_device(device)
        self.model.to(self.device)
        self.model.eval()

        self.generation_kwargs = {}
        if max_length is not None:
            self.generation_kwargs["max_length"] = max_length
        if num_beams is not None:
            self.generation_kwargs["num_beams"] = num_beams

    def predict(self, image: np.ndarray) -> str:
        return self.predict_batch([image])[0]

    def predict_batch(self, images: list[np.ndarray]) -> list[str]:
        if not images:
            return []

        pil_images = [self._to_rgb_image(image) for image in images]
        inputs = self.processor(images=pil_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values, **self.generation_kwargs)

        texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return [text.strip() for text in texts]

    def _resolve_device(self, device: str) -> str:
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available(")
        return device

    @staticmethod
    def _to_rgb_image(image: np.ndarray):
        from PIL import Image

        if image is None or image.size == 0:
            raise ValueError("на вход подалось пустое изображение")

        if image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.ndim == 3 and image.shape[2] == 4:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            raise ValueError(f"неподдерживаемый OCR размер изображения: {image.shape}")

        return Image.fromarray(rgb)