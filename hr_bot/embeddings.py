# embeddings.py
import os
import numpy as np
import requests
from functools import lru_cache
from typing import Optional
from sentence_transformers import SentenceTransformer

# --- Конфиг через переменные окружения ---
EMB_MODE = os.getenv("EMBEDDINGS_MODE", "local")  # local | openai
EMB_LOCAL_MODEL = os.getenv("EMBEDDINGS_LOCAL_MODEL", "all-MiniLM-L6-v2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


@lru_cache(maxsize=1)
def _get_local_model():
    model = SentenceTransformer(EMB_LOCAL_MODEL)
    return model, model.get_sentence_embedding_dimension()


def _normalize_vec(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / (norm + 1e-12)


def _embed_openai(text: str) -> np.ndarray:
    # text-embedding-3-small → 1536D
    dim = 1536
    if not text:
        return np.zeros(dim, dtype=float)
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {"model": "text-embedding-3-small", "input": text}
        resp = requests.post("https://api.openai.com/v1/embeddings", headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        vec = np.array(data["data"][0]["embedding"], dtype=float)
        return _normalize_vec(vec)
    except Exception as e:
        print(f"[OpenAI Embedding error]: {e}")
        return np.zeros(dim, dtype=float)


def embed_text(text: str) -> np.ndarray:
    """
    Единая функция эмбеддинга:
    - если EMB_MODE=local → SentenceTransformer
    - если EMB_MODE=openai → OpenAI embeddings
    """
    if EMB_MODE == "openai":
        return _embed_openai(text)

    # local
    model, dim = _get_local_model()
    if not text:
        return np.zeros(dim, dtype=float)
    emb = model.encode(text, convert_to_numpy=True)
    return _normalize_vec(emb)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def safe_float(value, default=0.0):
    try:
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
