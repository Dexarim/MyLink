# api_server.py
import os
import asyncio
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from llm_client import LLMClient
from scoring import base_similarity_score, detect_gaps
from qa_logic import generate_followup_question, integrate_followup

app = FastAPI(
    title="HR Match Analyzer API",
    description="Асинхронное API для анализа резюме и вакансий с помощью DeBERTa, эмбеддингов и LLM.",
    version="1.1.0",
)

# ✅ Разрешаем CORS-запросы от любых фронтов (можно ограничить позже)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # или ["http://localhost:3000"] если React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------- МОДЕЛИ ----------------------
class Vacancy(BaseModel):
    city: str
    experience: float
    post: str
    education: str
    languages: List[str]
    salary: float
    busyness: str


class Resume(BaseModel):
    city: str
    experience: float
    post: str
    education: str
    languages: List[str]
    salary: float
    busyness: str
    мотивация: Optional[str] = None
    skills: Optional[List[str]] = []


class FollowupAnswer(BaseModel):
    gap: Dict
    answer: str


# ---------------------- LLM CLIENT ----------------------
llm = LLMClient(
    provider=os.getenv("LLM_PROVIDER", "gemini"),
    model=os.getenv("LLM_MODEL", "gemini-1.5-flash"),
    api_key=os.getenv("GOOGLE_API_KEY"),
)


# ---------------------- ENDPOINTS ----------------------
@app.get("/")
async def index():
    return {
        "message": "🚀 HR Match Analyzer API активен",
        "endpoints": ["/analyze/base", "/analyze/followup", "/analyze/final"],
    }


@app.post("/analyze/base")
async def analyze_base(vacancy: Vacancy, resume: Resume):
    """
    Шаг 1️⃣ — базовая оценка схожести и определение несоответствий.
    """
    try:
        base = base_similarity_score(vacancy.dict(), resume.dict())
        gaps = detect_gaps(vacancy.dict(), resume.dict(), base)
        return {"base_score": base, "gaps": gaps}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/followup")
async def analyze_followup(vacancy: Vacancy, resume: Resume):
    """
    Шаг 2️⃣ — генерация уточняющих вопросов по найденным несовпадениям.
    """
    try:
        base = base_similarity_score(vacancy.dict(), resume.dict())
        gaps = detect_gaps(vacancy.dict(), resume.dict(), base)

        async def _generate_all():
            results = []
            for gap in gaps:
                q = await generate_followup_question(gap, vacancy.dict(), resume.dict(), llm)
                results.append({"gap": gap, "question": q})
            return results

        questions = await _generate_all()
        return {"questions": questions, "count": len(questions)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации вопросов: {e}")


@app.post("/analyze/final")
async def analyze_final(vacancy: Vacancy, resume: Resume, followups: List[FollowupAnswer]):
    """
    Шаг 3️⃣ — обработка ответов кандидата и пересчёт итогового скора.
    """
    try:
        base = base_similarity_score(vacancy.dict(), resume.dict())
        # выполнение CPU-bound задачи в отдельном потоке
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: integrate_followup(vacancy.dict(), resume.dict(), base, [f.dict() for f in followups])
        )
        return {"final_score": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка финального анализа: {e}")


# ---------------------- ЗАПУСК ----------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
