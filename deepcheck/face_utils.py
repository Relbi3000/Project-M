import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from typing import List


class FaceExtractor:
    def __init__(self, device: str | None = None, image_size: int = 224):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(keep_all=True, device=self.device, image_size=image_size)
        self.image_size = image_size

    def extract_faces_from_frame(self, frame_bgr) -> List[np.ndarray]:
        """Извлекает лица из кадра (NumPy BGR) и приводит их к размеру image_size x image_size."""
        if frame_bgr is None:
            return []
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        boxes, _ = self.mtcnn.detect(rgb)
        crops: List[np.ndarray] = []
        if boxes is None:
            return crops
        h, w, _ = rgb.shape
        for (x1, y1, x2, y2) in boxes:
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            face = rgb[y1:y2, x1:x2]
            if face.size == 0:
                continue
            face = cv2.resize(face, (self.image_size, self.image_size))
            crops.append(cv2.cvtColor(face, cv2.COLOR_RGB2BGR))  # возвращаем BGR
        return crops
