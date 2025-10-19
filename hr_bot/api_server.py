# api_server.py (версия с расширенной отладкой)
import os
import asyncio
import logging
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from dotenv import load_dotenv

# ---------------------- НАСТРОЙКА ЛОГОВ ----------------------
load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,  # Можно поставить INFO если слишком многословно
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("HR_API")

# ---------------------- ИМПОРТЫ ----------------------
from llm_client import LLMClient
from scoring import base_similarity_score, detect_gaps
from qa_logic import generate_followup_question, integrate_followup


app = FastAPI(
    title="HR Match Analyzer API",
    description="Асинхронное API для анализа резюме и вакансий с помощью DeBERTa, эмбеддингов и LLM.",
    version="1.1.1-debug",
)

from fastapi.middleware.cors import CORSMiddleware

# ---------------------- CORS ----------------------
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "*"  # можно убрать, если хочешь безопаснее
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
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
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.error("❌ GOOGLE_API_KEY отсутствует. Добавь его в .env!")
    raise RuntimeError("GOOGLE_API_KEY not found")

llm = LLMClient(
    provider=os.getenv("LLM_PROVIDER", "gemini"),
    model=os.getenv("LLM_MODEL", "gemini-1.5-flash-latest"),
    api_key=GOOGLE_API_KEY,
)

logger.info(f"✅ LLM клиент инициализирован: {llm.provider} / {llm.model}")


# ---------------------- MIDDLEWARE ----------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Логируем каждый запрос и ответ"""
    logger.info(f"➡️ {request.method} {request.url.path}")
    try:
        response = await call_next(request)
        logger.info(f"⬅️ {request.method} {request.url.path} → {response.status_code}")
        return response
    except Exception as e:
        logger.exception(f"💥 Ошибка при обработке {request.url.path}: {e}")
        raise


# ---------------------- ENDPOINTS ----------------------
@app.get("/")
async def index():
    logger.info("Запрос к корневому эндпоинту /")
    return {
        "message": "🚀 HR Match Analyzer API активен",
        "endpoints": ["/analyze/base", "/analyze/followup", "/analyze/final"],
    }


@app.post("/analyze/base")
async def analyze_base(vacancy: Vacancy, resume: Resume):
    """Базовая оценка схожести и определение несоответствий."""
    try:
        logger.debug(f"📄 Входные данные вакансии: {vacancy.dict()}")
        logger.debug(f"📄 Входные данные резюме: {resume.dict()}")

        base = base_similarity_score(vacancy.dict(), resume.dict())
        gaps = detect_gaps(vacancy.dict(), resume.dict(), base)

        logger.info(f"✅ Базовый скор: {base['base_score']}, найдено несоответствий: {len(gaps)}")
        return {"base_score": base, "gaps": gaps}

    except Exception as e:
        logger.exception("💥 Ошибка в /analyze/base:")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/followup")
async def analyze_followup(vacancy: Vacancy, resume: Resume):
    """Генерация уточняющих вопросов по найденным несовпадениям."""
    try:
        base = base_similarity_score(vacancy.dict(), resume.dict())
        gaps = detect_gaps(vacancy.dict(), resume.dict(), base)

        logger.info(f"🧩 Обнаружено {len(gaps)} несовпадений, начинаю генерацию уточняющих вопросов...")

        async def _generate_all():
            results = []
            for gap in gaps:
                logger.debug(f"🤖 Генерация вопроса по gap: {gap}")
                q = await generate_followup_question(gap, vacancy.dict(), resume.dict(), llm)
                results.append({"gap": gap, "question": q})
            return results

        questions = await _generate_all()
        logger.info(f"✅ Сгенерировано вопросов: {len(questions)}")
        return {"questions": questions, "count": len(questions)}

    except Exception as e:
        logger.exception("💥 Ошибка в /analyze/followup:")
        raise HTTPException(status_code=500, detail=f"Ошибка генерации вопросов: {e}")


@app.post("/analyze/final")
async def analyze_final(vacancy: Vacancy, resume: Resume, followups: List[FollowupAnswer]):
    """Обработка ответов кандидата и пересчёт итогового скора."""
    try:
        logger.info("⚙️ Финальный анализ кандидата...")
        base = base_similarity_score(vacancy.dict(), resume.dict())

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: integrate_followup(vacancy.dict(), resume.dict(), base, [f.dict() for f in followups])
        )

        logger.info(f"✅ Итоговый скор после уточнений: {result}")
        return {"final_score": result}

    except Exception as e:
        logger.exception("💥 Ошибка в /analyze/final:")
        raise HTTPException(status_code=500, detail=f"Ошибка финального анализа: {e}")


# ---------------------- ЗАПУСК ----------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Запуск HR Match Analyzer API...")
    uvicorn.run("api_server:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
