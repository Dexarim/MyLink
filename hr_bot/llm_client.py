import os
import json
import httpx
import asyncio
from functools import lru_cache
from typing import Optional

DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
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
    async def generate(self, prompt: str) -> str:
        """
        Асинхронная генерация текста для FastAPI.
        """
        if self.provider == "gemini":
            return await self._generate_gemini(prompt)
        elif self.provider in ("ollama", "mistral"):
            return await self._generate_ollama(prompt)
        return "[LLM error] Unknown provider"

    async def _generate_gemini(self, prompt: str) -> str:
        """
        Асинхронный вызов Gemini 1.5 Flash через REST API (v1beta)
        С задержкой для стабилизации сетевого обмена.
        """
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"

            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key
            }

            payload = {
                "contents": [
                    {"parts": [{"text": prompt}]}
                ]
            }

            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(url, headers=headers, json=payload)
                await asyncio.sleep(0.3)  # ⏳ задержка 300 мс (стабилизация)
                resp.raise_for_status()

            result = resp.json()
            text = (
                result.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
                .strip()
            )
            return text or "[Gemini: пустой ответ]"

        except httpx.HTTPStatusError as e:
            return f"[Gemini error]: HTTP {e.response.status_code} - {e.response.text}"
        except Exception as e:
            return f"[Gemini error]: {e}"

    async def _generate_ollama(self, prompt: str) -> str:
        """
        Асинхронный вызов Ollama / Mistral API
        """
        try:
            url = f"{self.base_url}/api/generate"
            payload = {"model": self.model, "prompt": prompt}
            text = ""

            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", url, json=payload) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_text():
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        text += data.get("response", "")

            return text.strip()

        except httpx.HTTPStatusError as e:
            return f"[Ollama error]: HTTP {e.response.status_code} - {e.response.text}"
        except Exception as e:
            return f"[Ollama error]: {e}"
