# deberta_analyzer.py
import os
from typing import Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

DEFAULT_MODEL = os.getenv("DEBERTA_MODEL", "microsoft/deberta-v3-base-mnli")

_tokenizer = None
_model = None
_pipe = None


def _ensure_pipeline():
    global _tokenizer, _model, _pipe
    if _pipe is not None:
        return _pipe
    device = 0 if torch.cuda.is_available() else -1
    _tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    _model = AutoModelForSequenceClassification.from_pretrained(DEFAULT_MODEL)
    _pipe = pipeline("text-classification", model=_model, tokenizer=_tokenizer, device=device)
    return _pipe


def analyze_text(resume_text: str, job_description: str) -> Dict:
    """
    Грубая оценка соответствия через классификационную голову DeBERTa (MNLI-вариант предпочтителен).
    Возвращает label и confidence (score).
    """
    pipe = _ensure_pipeline()
    prompt = (
        "Оцени, подходит ли кандидат для должности. "
        f"Резюме: {resume_text}\n"
        f"Описание должности: {job_description}"
    )
    out = pipe(prompt, truncation=True)
    if isinstance(out, list) and out:
        label = out[0].get("label", "LABEL_0")
        score = float(out[0].get("score", 0.0))
        mapped = "Подходит" if "1" in label or "POS" in label or "ENTAIL" in label else "Не подходит"
        return {"label": mapped, "raw_label": label, "confidence": round(score, 3)}
    return {"label": "Неопределено", "raw_label": "", "confidence": 0.0}
