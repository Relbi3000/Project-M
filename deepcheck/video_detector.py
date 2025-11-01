import cv2
import numpy as np
from pathlib import Path
from .face_utils import FaceExtractor
from .heuristics import face_deepfake_score

class VideoDeepfakeDetector:
    def __init__(self, device: str | None = None, fps_sample: float = 2.0):
        self.face_extractor = FaceExtractor(device=device, image_size=224)
        self.fps_sample = fps_sample

    def predict_file(self, video_path: str) -> dict:
        p = Path(video_path)
        if not p.exists():
            raise FileNotFoundError(video_path)

        cap = cv2.VideoCapture(str(p))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(fps / self.fps_sample))

        frame_idx = 0
        face_scores = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                faces = self.face_extractor.extract_faces_from_frame(frame)
                for f in faces:
                    face_scores.append(face_deepfake_score(f))
            frame_idx += 1

        cap.release()

        if not face_scores:
            return {
                "type": "video",
                "path": str(p),
                "label": "unknown",
                "probabilities": {"deepfake": 0.0, "real": 0.0},
                "note": "Лица не найдены — невозможно оценить."
            }

        # агрегируем — возьмём перцентиль 75 как устойчивую оценку «худшего» случая
        s = float(np.percentile(face_scores, 75))
        prob_df = s
        prob_real = 1.0 - prob_df
        label = "deepfake" if prob_df >= 0.6 else ("uncertain" if prob_df >= 0.4 else "real")

        return {
            "type": "video",
            "path": str(p),
            "label": label,
            "probabilities": {"deepfake": prob_df, "real": prob_real},
            "faces_scored": len(face_scores)
        }
