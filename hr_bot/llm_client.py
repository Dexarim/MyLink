# llm_client.py
import os
import json
import requests
from functools import lru_cache
from typing import Optional

DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # gemini | ollama | mistral (ollama)
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


class LLMClient:
    def __init__(
        self,
        provider: str = DEFAULT_PROVIDER,
        model: str = DEFAULT_MODEL,
        base_url: str = OLLAMA_URL,
        api_key: Optional[str] = GOOGLE_API_KEY,
    ):
        self.provider = provider
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    @lru_cache(maxsize=128)
    def generate(self, prompt: str) -> str:
        if self.provider == "gemini":
            return self._generate_gemini(prompt)
        # и mistral, и ollama идут через один API ollama
        elif self.provider in ("ollama", "mistral"):
            return self._generate_ollama(prompt)
        return "[LLM error] Unknown provider"

    def _generate_gemini(self, prompt: str) -> str:
        try:
            url = f"https://generativelanguage.googleapis.com/v1/models/{self.model}:generateContent?key={self.api_key}"
            headers = {"Content-Type": "application/json"}
            data = {
                "contents": [
                    {"role": "user", "parts": [{"text": prompt}]}
                ]
            }
            resp = requests.post(url, headers=headers, json=data, timeout=30)
            resp.raise_for_status()
            result = resp.json()
            return result["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            return f"[Gemini error]: {e}"

    def _generate_ollama(self, prompt: str) -> str:
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
            return f"[Ollama error]: {e}"
