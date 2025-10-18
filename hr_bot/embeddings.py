import numpy as np
from typing import Dict, List, Any
from sentence_transformers import SentenceTransformer, util


EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
EMB_DIM = EMB_MODEL.get_sentence_embedding_dimension()

def normalize_vec(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / (norm + 1e-12)


def embed_text(text: str) -> np.ndarray:
    if not text:
        return np.zeros(EMB_DIM, dtype=float)
    emb = EMB_MODEL.encode(text, convert_to_numpy=True)
    return normalize_vec(emb)

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
