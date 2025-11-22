import numpy as np
import cv2


def fft_highfreq_energy(img_bgr: np.ndarray) -> float:
    """Оценивает долю высокочастотной энергии изображения (простая метрика резкости/шума)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    h, w = magnitude.shape
    # Убираем низкие частоты в центре, считаем долю высокого спектра
    c = 16
    center = magnitude[h // 2 - c:h // 2 + c, w // 2 - c:w // 2 + c]
    total = magnitude.sum() + 1e-6
    low = center.sum()
    high = total - low
    return float(high / total)


def jpeg_blockiness(img_bgr: np.ndarray) -> float:
    """Блочность JPEG [0..1]: доля резких границ сетки 8x8 относительно общего градиента."""
    if img_bgr is None or img_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_mean = float(np.mean(np.abs(gx)) + np.mean(np.abs(gy)) + 1e-6)
    diffs_v = []
    for x in range(8, gray.shape[1], 8):
        diffs_v.append(np.abs(gray[:, x].astype(np.float32) - gray[:, x - 1].astype(np.float32)))
    diffs_h = []
    for y in range(8, gray.shape[0], 8):
        diffs_h.append(np.abs(gray[y, :].astype(np.float32) - gray[y - 1, :].astype(np.float32)))
    if not diffs_v and not diffs_h:
        return 0.0
    edge_mean = 0.0
    if diffs_v:
        edge_mean += float(np.mean(diffs_v))
    if diffs_h:
        edge_mean += float(np.mean(diffs_h))
    edge_mean /= (2 if diffs_v and diffs_h else 1)
    b = edge_mean / (edge_mean + grad_mean)
    return float(np.clip(b, 0.0, 1.0))


def mouth_eye_inconsistency(face_bgr: np.ndarray) -> float:
    """Измеряет несогласованность текстуры в верхней (глаза) и нижней (рот) частях лица."""
    h, w, _ = face_bgr.shape
    top = face_bgr[int(0.2 * h):int(0.4 * h), int(0.2 * w):int(0.8 * w)]
    bottom = face_bgr[int(0.6 * h):int(0.85 * h), int(0.2 * w):int(0.8 * w)]

    def blockiness(x):
        if x.size == 0:
            return 0.0
        gx = cv2.Sobel(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), cv2.CV_32F, 0, 1, ksize=3)
        return float(np.mean(np.abs(gx)) + np.mean(np.abs(gy)))

    return (blockiness(top) + blockiness(bottom)) / 2.0


def face_deepfake_score(face_bgr: np.ndarray) -> float:
    """Эмпирическая оценка вероятности дипфейка по простым признакам [0..1]."""
    hf = fft_highfreq_energy(face_bgr)          # [0..1+] — доля высоких частот
    inc = mouth_eye_inconsistency(face_bgr)     # расхождение текстур
    hf_n = min(1.0, hf * 2.0)
    inc_n = 1 - np.exp(-inc / 30.0)             # 0..1
    return float(np.clip(0.6 * hf_n + 0.4 * inc_n, 0.0, 1.0))


def laplacian_sharpness(img_bgr: np.ndarray) -> float:
    """Дисперсия лапласиана как метрика резкости кадра."""
    if img_bgr is None or img_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def face_deepfake_score_blocky(face_bgr: np.ndarray) -> float:
    """Корректировка face_deepfake_score с учётом блочности JPEG, чтобы снизить ложные срабатывания."""
    s0 = face_deepfake_score(face_bgr)
    b = jpeg_blockiness(face_bgr)
    adj = s0 * (1.0 - 0.35 * b)
    return float(np.clip(adj, 0.0, 1.0))
