# api_server.py
import os
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import JSONResponse

from llm_client import LLMClient
from scoring import base_similarity_score, detect_gaps
from qa_logic import generate_followup_question
from qa_logic import integrate_followup
from deberta_analyzer import analyze_text
from embeddings import embed_text, cosine_sim

app = FastAPI(title="HR Analyzer API", version="1.0.0")

# === Инициализация LLM ===
LLM = LLMClient(
    provider=os.getenv("LLM_PROVIDER", "gemini"),
    model=os.getenv("LLM_MODEL", "gemini-1.5-flash"),
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
)


@app.post("/llm/")
async def llm_generate(payload: dict = Body(...)):
    try:
        prompt = payload.get("prompt", "")
        if not prompt:
            raise HTTPException(400, "prompt is required")
        return {"response": LLM.generate(prompt)}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/embeddings/similarity/")
async def embeddings_similarity(payload: dict = Body(...)):
    try:
        t1 = payload.get("text1", "")
        t2 = payload.get("text2", "")
        sim = cosine_sim(embed_text(t1), embed_text(t2))
        return {"similarity": sim}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/ml/deberta/")
async def ml_deberta(payload: dict = Body(...)):
    try:
        resume_text = payload.get("resume", "")
        job_text = payload.get("vacancy", "")
        result = analyze_text(resume_text, job_text)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/score/")
async def score(payload: dict = Body(...)):
    try:
        vacancy = payload["vacancy"]
        resume  = payload["resume"]
        base = base_similarity_score(vacancy, resume)
        return base
    except KeyError:
        raise HTTPException(400, "payload must contain 'vacancy' and 'resume'")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/analyze/")
async def analyze_full(payload: dict = Body(...)):
    """
    Комбо-эндпоинт:
    - базовый скоринг
    - детект гэпов
    - генерация вопросов (LLM/шаблоны)
    - DeBERTa-анализ соответствия
    Возвращает всё в одном JSON.
    """
    try:
        vacancy = payload["vacancy"]
        resume  = payload["resume"]

        # 1) базовый скоринг
        base = base_similarity_score(vacancy, resume)

        # 2) гэпы
        gaps = detect_gaps(vacancy, resume, base)

        # 3) вопросы
        questions = [generate_followup_question(g, vacancy, resume, LLM) for g in gaps]

        # 4) DeBERTa
        deberta_result = analyze_text(str(resume), str(vacancy))

        # 5) итоговая рекомендация (простая логика)
        rec = "Кандидат не подходит"
        score_val = base["base_score"]
        if score_val >= 75:
            rec = "Кандидат подходит"
        elif score_val >= 50:
            rec = "Возможен найм после обучения/уточнений"

        return JSONResponse({
            "base_score": base["base_score"],
            "details": base["details"],
            "gaps": gaps,
            "questions": questions,
            "deberta_result": deberta_result,
            "recommendation": rec
        })
    except KeyError:
        raise HTTPException(400, "payload must contain 'vacancy' and 'resume'")
    except Exception as e:
        raise HTTPException(500, str(e))
