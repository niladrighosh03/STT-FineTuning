import base64
from typing import Any, Dict

import requests


class RunpodSTTService:
    def __init__(self, endpoint_id: str, api_key: str, language: str = "en"):
        self.endpoint_id = endpoint_id
        self.api_key = api_key
        self.language = language
        self.base_url = f"https://{endpoint_id}.api.runpod.ai"

    def transcribe_file(self, audio_path: str) -> Dict[str, Any]:
        with open(audio_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        response = requests.post(
            f"{self.base_url}/transcribe",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "audio_base64": audio_base64,
                "language": self.language,
            },
            timeout=180,
        )
        response.raise_for_status()
        return response.json()

    def transcribe_text(self, audio_path: str) -> str:
        result = self.transcribe_file(audio_path)
        return result["DisplayText"]
