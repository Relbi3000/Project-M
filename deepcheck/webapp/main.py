from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

# Supported extensions (keep in sync with deepcheck.cli)
AUDIO_EXT = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


app = FastAPI(title="Deepcheck Web UI", docs_url=None, redoc_url=None)

_here = Path(__file__).resolve().parent
static_dir = _here / "static"
templates = Jinja2Templates(directory=str(_here / "templates"))
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


def _save_upload_to_tmp(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "upload").suffix
    fd, tmp_path = tempfile.mkstemp(prefix="deepcheck_", suffix=suffix)
    os.close(fd)
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return Path(tmp_path)


def _media_type_from_path(p: Path) -> Optional[str]:
    ext = p.suffix.lower()
    if ext in AUDIO_EXT:
        return "audio"
    if ext in VIDEO_EXT:
        return "video"
    if ext in IMG_EXT:
        return "image"
    return None


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, file: UploadFile = File(...)):
    filename = file.filename or "upload"
    tmp_path: Optional[Path] = None
    error: Optional[str] = None
    result: Optional[dict] = None

    try:
        try:
            form = await request.form()
            sensitivity = str(form.get("sensitivity") or "normal").lower()
        except Exception:
            sensitivity = "normal"

        agg, perc, df_th, un_th = _params_for_sensitivity(sensitivity)
        tmp_path = _save_upload_to_tmp(file)
        mtype = _media_type_from_path(tmp_path)
        if not mtype:
            error = f"Неподдерживаемое расширение файла: {Path(filename).suffix}"
        else:
            if mtype == "video":
                try:
                    from deepcheck.video_detector import VideoDeepfakeDetector

                    det = VideoDeepfakeDetector(
                        aggregate=agg,
                        percentile=perc,
                        df_threshold=df_th,
                        uncertain_threshold=un_th,
                    )
                    result = det.predict_file(str(tmp_path))
                except Exception as e:
                    error = f"Ошибка при анализе видео: {e}"
            elif mtype == "image":
                try:
                    import cv2
                    from deepcheck.face_utils import FaceExtractor
                    from deepcheck.heuristics import face_deepfake_score_blocky, laplacian_sharpness

                    img = cv2.imread(str(tmp_path))
                    faces = FaceExtractor().extract_faces_from_frame(img)
                    faces = [f for f in faces if laplacian_sharpness(f) >= 30.0]
                    if not faces:
                        result = {
                            "type": "image",
                            "path": str(filename),
                            "label": "unknown",
                            "probabilities": {"deepfake": 0.0, "real": 0.0},
                            "note": "Лицо не обнаружено или кадр слишком размытый.",
                        }
                    else:
                        scores = [face_deepfake_score_blocky(f) for f in faces]
                        try:
                            import numpy as _np

                            s = float(_np.median(scores))
                        except Exception:
                            s = max(scores)
                        if s >= df_th:
                            label = "deepfake"
                        elif s >= un_th:
                            label = "uncertain"
                        else:
                            label = "real"
                        result = {
                            "type": "image",
                            "path": str(filename),
                            "label": label,
                            "probabilities": {"deepfake": float(s), "real": float(1.0 - s)},
                            "faces_scored": len(scores),
                        }
                except Exception as e:
                    error = f"Ошибка при анализе изображения: {e}"
            elif mtype == "audio":
                try:
                    from deepcheck.audio_detector import AudioDeepfakeDetector

                    det = AudioDeepfakeDetector()
                    result = det.predict_file(str(tmp_path))
                except Exception as e:
                    error = (
                        "Ошибка при анализе аудио: "
                        + str(e)
                        + ". Чаще всего это означает, что не скачаны веса AASIST."
                    )

    finally:
        try:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    explanations = _build_explanations(result)
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "filename": filename,
            "error": error,
            "result": result,
            "explanations": explanations,
        },
    )


def _build_explanations(result: Optional[dict]) -> dict:
    if not result:
        return {}

    label = result.get("label")
    label_explain = {
        "deepfake": "Высокая вероятность синтетического контента (deepfake).",
        "uncertain": "Сигнал неоднозначный; лучше перепроверить вручную.",
        "real": "Признаков синтетики не найдено; материал выглядит натуральным.",
        "unknown": "Не удалось проанализировать материал.",
    }.get(label, "")

    prob_df = None
    try:
        prob_df = float(result.get("probabilities", {}).get("deepfake", 0.0))
    except Exception:
        prob_df = None

    confidence_text = None
    if prob_df is not None:
        if prob_df >= 0.85:
            confidence_text = "Уверенность высокая."
        elif prob_df >= 0.6:
            confidence_text = "Уверенность умеренная; при необходимости перепроверьте вручную."
        elif prob_df >= 0.4:
            confidence_text = "Сигнал слабый; воспринимайте как предупреждение."
        else:
            confidence_text = "Модель считает материал натуральным."

    notes = []
    if result.get("type") in {"video", "image"} and result.get("faces_scored", 0) == 0:
        notes.append("Нужны кадры с лицом; без лица модель не умеет оценивать.")

    if result.get("type") == "audio" and "score_raw" in result:
        notes.append("score_raw — сырое значение AASIST; используйте probabilities для решения.")

    return {
        "label": label_explain,
        "confidence": confidence_text,
        "notes": notes,
        "probabilities": {
            "deepfake": "Вероятность синтетики по модели (0..1).",
            "real": "1 - deepfake.",
        },
        "faces_scored": "Сколько лиц было проанализировано (после фильтра по резкости).",
        "type": "Тип входного файла (audio/video/image).",
        "path": "Имя загруженного файла.",
    }


def _params_for_sensitivity(s: str) -> tuple[str, float, float, float]:
    s = (s or "normal").lower().strip()
    if s == "conservative":
        return ("median", 65.0, 0.8, 0.6)
    if s == "aggressive":
        return ("percentile", 55.0, 0.6, 0.4)
    return ("median", 60.0, 0.7, 0.5)
