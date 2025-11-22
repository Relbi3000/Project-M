import argparse
from pathlib import Path

from .api_fallback import CheapApiFallback

AUDIO_EXT = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    parser = argparse.ArgumentParser(description="Deepfake/Deepvoice detector (MVP).")
    parser.add_argument("path", help="Путь к файлу (audio/video/image).")
    parser.add_argument(
        "--api-fallback",
        action="store_true",
        help="Попробовать внешний API при сомнительной уверенности (если задан DEEPCHECK_API_KEY).",
    )
    parser.add_argument(
        "--low-threshold",
        type=float,
        default=0.4,
        help="Нижний порог уверенности для запроса внешнего API.",
    )
    parser.add_argument("--df-threshold", type=float, default=0.7, help="Порог для метки deepfake.")
    parser.add_argument(
        "--uncertain-threshold",
        type=float,
        default=0.5,
        help="Порог для метки uncertain.",
    )
    parser.add_argument(
        "--video-aggregate",
        choices=["median", "percentile"],
        default="median",
        help="Агрегирование кадров: медиана или процентиль.",
    )
    parser.add_argument(
        "--video-percentile",
        type=float,
        default=60.0,
        help="Процентиль, если выбран percentile.",
    )
    args = parser.parse_args()

    p = Path(args.path)
    if not p.exists():
        raise SystemExit(f"Файл не найден: {p}")

    ext = p.suffix.lower()
    result = None

    if ext in AUDIO_EXT:
        try:
            from .audio_detector import AudioDeepfakeDetector
        except Exception as e:
            raise SystemExit(f"Не удалось инициализировать аудио-модель: {e}")
        det = AudioDeepfakeDetector()
        result = det.predict_file(str(p))
    elif ext in VIDEO_EXT:
        try:
            from .video_detector import VideoDeepfakeDetector
        except Exception as e:
            raise SystemExit(f"Не удалось инициализировать видео-модель: {e}")
        det = VideoDeepfakeDetector(
            aggregate=args.video_aggregate,
            percentile=args.video_percentile,
            df_threshold=args.df_threshold,
            uncertain_threshold=args.uncertain_threshold,
        )
        result = det.predict_file(str(p))
    elif ext in IMG_EXT:
        import cv2
        from .face_utils import FaceExtractor
        from .heuristics import face_deepfake_score_blocky, laplacian_sharpness

        img = cv2.imread(str(p))
        faces = FaceExtractor().extract_faces_from_frame(img)
        faces = [f for f in faces if laplacian_sharpness(f) >= 30.0]
        if not faces:
            result = {
                "type": "image",
                "path": str(p),
                "label": "unknown",
                "probabilities": {"deepfake": 0.0, "real": 0.0},
                "note": "Лицо не обнаружено.",
            }
        else:
            scores = [face_deepfake_score_blocky(f) for f in faces]
            try:
                import numpy as _np

                s = float(_np.median(scores))
            except Exception:
                s = max(scores)
            if s >= args.df_threshold:
                label = "deepfake"
            elif s >= args.uncertain_threshold:
                label = "uncertain"
            else:
                label = "real"
            result = {
                "type": "image",
                "path": str(p),
                "label": label,
                "probabilities": {"deepfake": float(s), "real": float(1.0 - s)},
                "faces_scored": len(scores),
            }
    else:
        raise SystemExit(f"Неподдерживаемое расширение: {ext}")

    if (
        args.api_fallback
        and result.get("probabilities", {}).get("deepfake", 0.0) >= args.low_threshold
        and result.get("probabilities", {}).get("deepfake", 0.0) <= 0.7
    ):
        api = CheapApiFallback()
        api_res = api.check_media(str(p), media_type=result["type"])
        result["api_fallback"] = api_res

    import json

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
