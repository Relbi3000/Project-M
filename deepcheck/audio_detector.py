import torch
from pathlib import Path

# AASIST из speechbrain (скачает веса с HF при первом запуске)
from speechbrain.inference.AntiSpoof import AntiSpoof

class AudioDeepfakeDetector:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Сохранит веса локально в папке pretrained_models/
        self.model = AntiSpoof.from_hparams(
            source="speechbrain/antispoofing-AASIST",
            savedir="pretrained_models/antispoof_AASIST",
            run_opts={"device": self.device}
        )

    @torch.inference_mode()
    def predict_file(self, audio_path: str) -> dict:
        p = Path(audio_path)
        if not p.exists():
            raise FileNotFoundError(audio_path)
        # Модель возвращает score и метку "bonafide/spoof"
        score, prediction = self.model.classify_file(str(p))
        # Приведём к удобному словарю
        # Чем выше score — тем более "bonafide" (честная речь).
        # Дадим вероятность "spoof" как (1 - sigmoid(score)) для наглядности.
        prob_bonafide = torch.sigmoid(torch.tensor(float(score))).item()
        prob_spoof = 1.0 - prob_bonafide
        return {
            "type": "audio",
            "path": str(p),
            "label": "spoof" if prediction == "spoof" else "bonafide",
            "score_raw": float(score),
            "probabilities": {"bonafide": prob_bonafide, "spoof": prob_spoof}
        }
