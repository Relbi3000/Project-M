import argparse
from pathlib import Path

# Импорт детекторов откладываем до момента использования, чтобы `--help`
# и базовые операции работали без установки всех тяжёлых зависимостей.
from .api_fallback import CheapApiFallback

AUDIO_EXT = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}
IMG_EXT   = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    parser = argparse.ArgumentParser(description="Deepfake/Deepvoice detector (MVP).")
    parser.add_argument("path", help="Путь к файлу медиа")
    parser.add_argument("--api-fallback", action="store_true", help="Задействовать внешнее API при низкой уверенностью")
    parser.add_argument("--low-threshold", type=float, default=0.4, help="Порог неуверенности локальной модели")
    args = parser.parse_args()

    p = Path(args.path)
    if not p.exists():
        raise SystemExit(f"Файл не найден: {p}")

    ext = p.suffix.lower()
    result = None

    if ext in AUDIO_EXT:
        # ленивый импорт, чтобы не падать при проблемах с torchaudio/speechbrain
        try:
            from .audio_detector import AudioDeepfakeDetector
        except Exception as e:
            raise SystemExit(f"Не удалось загрузить модуль аудио-детекции: {e}")
        det = AudioDeepfakeDetector()
        result = det.predict_file(str(p))
    elif ext in VIDEO_EXT:
        try:
            from .video_detector import VideoDeepfakeDetector
        except Exception as e:
            raise SystemExit(f"Не удалось загрузить модуль видео-детекции: {e}")
        det = VideoDeepfakeDetector()
        result = det.predict_file(str(p))
    elif ext in IMG_EXT:
        # Обработка одиночного изображения той же эвристикой
        import cv2
        from .face_utils import FaceExtractor
        from .heuristics import face_deepfake_score
        img = cv2.imread(str(p))
        faces = FaceExtractor().extract_faces_from_frame(img)
        if not faces:
            result = {
                "type": "image", "path": str(p), "label": "unknown",
                "probabilities": {"deepfake": 0.0, "real": 0.0},
                "note": "Лица не найдены"
            }
        else:
            scores = [face_deepfake_score(f) for f in faces]
            s = max(scores)
            result = {
                "type": "image", "path": str(p),
                "label": "deepfake" if s >= 0.6 else ("uncertain" if s >= 0.4 else "real"),
                "probabilities": {"deepfake": float(s), "real": float(1.0 - s)},
                "faces_scored": len(scores)
            }
    else:
        raise SystemExit(f"Неподдерживаемое расширение: {ext}")

    # при необходимости — «второе мнение» дешёвого API
    if args.api_fallback and result.get("probabilities", {}).get("deepfake", 0.0) >= args.low_threshold and \
       result.get("probabilities", {}).get("deepfake", 0.0) <= 0.7:
        api = CheapApiFallback()
        api_res = api.check_media(str(p), media_type=result["type"])
        result["api_fallback"] = api_res

    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
