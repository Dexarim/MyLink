import json
import requests
from functools import lru_cache


# === Клиент для Mistral/Ollama ===
class MistralClient:
    def __init__(self, model="mistral", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    @lru_cache(maxsize=128)
    def generate(self, prompt: str) -> str:
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt},
                stream=True,
                timeout=30
            )
            text = ""
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line.decode("utf-8"))
                text += data.get("response", "")
            return text.strip()
        except Exception as e:
            return f"[Ошибка Mistral: {e}]"