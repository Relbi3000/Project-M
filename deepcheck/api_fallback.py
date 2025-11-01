import os
import time
from typing import Literal, Optional

class CheapApiFallback:
    """
    Заглушка-обёртка: сюда добавим реальный провайдер (HTTP POST),
    чтобы дергать его ТОЛЬКО когда локальная модель не уверена.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("DEEPCHECK_API_KEY")

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def check_media(self, file_path: str, media_type: Literal["audio", "video", "image"]) -> dict:
        if not self.is_configured():
            return {"used": False, "reason": "no_api_key"}
        # Здесь будет настоящая отправка файла на API и разбор ответа.
        # Пока — имитируем ответ.
        time.sleep(0.2)
        return {
            "used": True,
            "provider": "placeholder",
            "label": "deepfake",
            "confidence": 0.7
        }
