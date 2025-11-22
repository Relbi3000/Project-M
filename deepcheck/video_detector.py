import cv2
import numpy as np
from pathlib import Path

from .face_utils import FaceExtractor
from .heuristics import face_deepfake_score_blocky, laplacian_sharpness


class VideoDeepfakeDetector:
    def __init__(
        self,
        device: str | None = None,
        fps_sample: float = 2.0,
        aggregate: str = "median",
        percentile: float = 60.0,
        df_threshold: float = 0.8,
        uncertain_threshold: float = 0.6,
        min_sharpness: float = 50.0,
        strong_threshold: float = 0.9,
        min_high_ratio: float = 0.3,
        min_faces_considered: int = 12,
    ):
        self.face_extractor = FaceExtractor(device=device, image_size=224)
        self.fps_sample = fps_sample
        self.aggregate = aggregate
        self.percentile = percentile
        self.df_threshold = df_threshold
        self.uncertain_threshold = uncertain_threshold
        self.min_sharpness = min_sharpness
        self.strong_threshold = strong_threshold
        self.min_high_ratio = min_high_ratio
        self.min_faces_considered = min_faces_considered

    def predict_file(self, video_path: str) -> dict:
        p = Path(video_path)
        if not p.exists():
            raise FileNotFoundError(video_path)

        cap = cv2.VideoCapture(str(p))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(fps / self.fps_sample))

        frame_idx = 0
        face_scores: list[float] = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                faces = self.face_extractor.extract_faces_from_frame(frame)
                for f in faces:
                    if laplacian_sharpness(f) < self.min_sharpness:
                        continue
                    face_scores.append(face_deepfake_score_blocky(f))
            frame_idx += 1

        cap.release()

        if not face_scores:
            return {
                "type": "video",
                "path": str(p),
                "label": "unknown",
                "probabilities": {"deepfake": 0.0, "real": 0.0},
                "note": "Лица не обнаружены, оценка не выполнена.",
            }

        # Агрегируем оценки лиц по медиане или процентилю
        if self.aggregate == "percentile":
            s = float(np.percentile(face_scores, self.percentile))
        else:
            s = float(np.median(face_scores))
        prob_df = s
        prob_real = 1.0 - prob_df

        strong = sum(1 for v in face_scores if v >= self.strong_threshold)
        ratio = strong / len(face_scores)

        if len(face_scores) < self.min_faces_considered and prob_df >= self.df_threshold:
            label = "uncertain"
        elif prob_df >= self.df_threshold and ratio >= self.min_high_ratio:
            label = "deepfake"
        elif prob_df >= self.uncertain_threshold:
            label = "uncertain"
        else:
            label = "real"

        return {
            "type": "video",
            "path": str(p),
            "label": label,
            "probabilities": {"deepfake": prob_df, "real": prob_real},
            "faces_scored": len(face_scores),
            "strong_support": {"count": strong, "ratio": float(ratio)},
        }
