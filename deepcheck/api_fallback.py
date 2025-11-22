import os
import time
from typing import Literal, Optional


class CheapApiFallback:
    """
    Заглушка для внешнего API: имитирует проверку файла у стороннего провайдера,
    если указан DEEPCHECK_API_KEY.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("DEEPCHECK_API_KEY")

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def check_media(self, file_path: str, media_type: Literal["audio", "video", "image"]) -> dict:
        if not self.is_configured():
            return {"used": False, "reason": "no_api_key"}
        # Здесь была бы отправка файла в API; оставляем имитацию ответа
        time.sleep(0.2)
        return {
            "used": True,
            "provider": "placeholder",
            "label": "deepfake",
            "confidence": 0.7,
        }
