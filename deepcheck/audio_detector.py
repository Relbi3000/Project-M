from pathlib import Path

import torch
from speechbrain.inference.AntiSpoof import AntiSpoof


class AudioDeepfakeDetector:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Весы автоматически скачаются в pretrained_models/
        self.model = AntiSpoof.from_hparams(
            source="speechbrain/antispoofing-AASIST",
            savedir="pretrained_models/antispoof_AASIST",
            run_opts={"device": self.device},
        )

    @torch.inference_mode()
    def predict_file(self, audio_path: str) -> dict:
        p = Path(audio_path)
        if not p.exists():
            raise FileNotFoundError(audio_path)

        score, prediction = self.model.classify_file(str(p))
        # Сырой score относится к "bonafide/spoof"; применяем sigmoid для вероятностей
        prob_bonafide = torch.sigmoid(torch.tensor(float(score))).item()
        prob_spoof = 1.0 - prob_bonafide
        return {
            "type": "audio",
            "path": str(p),
            "label": "spoof" if prediction == "spoof" else "bonafide",
            "score_raw": float(score),
            "probabilities": {"bonafide": prob_bonafide, "spoof": prob_spoof},
        }
