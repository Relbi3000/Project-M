import numpy as np
import cv2

def fft_highfreq_energy(img_bgr: np.ndarray) -> float:
    """Оцениваем долю высоких частот (артефакты генерации)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    h, w = magnitude.shape
    # вырежем центральную (низкочастотную) область
    c = 16
    center = magnitude[h//2-c:h//2+c, w//2-c:w//2+c]
    total = magnitude.sum() + 1e-6
    low = center.sum()
    high = total - low
    return float(high / total)


def mouth_eye_inconsistency(face_bgr: np.ndarray) -> float:
    """Грубая эвристика: нестабильность текстур вокруг рта/глаз (блоковая/шум)."""
    h, w, _ = face_bgr.shape
    # зоны: глаза (верх), рот (низ)
    top = face_bgr[int(0.2*h):int(0.4*h), int(0.2*w):int(0.8*w)]
    bottom = face_bgr[int(0.6*h):int(0.85*h), int(0.2*w):int(0.8*w)]
    def blockiness(x):
        if x.size == 0:
            return 0.0
        gx = cv2.Sobel(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), cv2.CV_32F, 0, 1, ksize=3)
        return float(np.mean(np.abs(gx)) + np.mean(np.abs(gy)))
    return (blockiness(top) + blockiness(bottom)) / 2.0


def face_deepfake_score(face_bgr: np.ndarray) -> float:
    """Соберём простую метрику [0..1] — выше = вероятнее дипфейк."""
    hf = fft_highfreq_energy(face_bgr)          # [0..1+] — нормируем мягко
    inc = mouth_eye_inconsistency(face_bgr)     # произвольная шкала
    # нормализация/сигмоида
    hf_n = min(1.0, hf * 2.0)
    inc_n = 1 - np.exp(-inc / 30.0)             # плавная 0..1
    return float(np.clip(0.6 * hf_n + 0.4 * inc_n, 0.0, 1.0))
