"""
Интерактивная версия системы релевантности с follow-up (Mistral/Ollama + эмбеддинги)
После анализа резюме бот задаёт вопросы в консоли, а пользователь отвечает вручную.
"""

import json
import requests
import numpy as np
from typing import Dict, List, Any
from functools import lru_cache
from sentence_transformers import SentenceTransformer, util

# === Инициализация модели эмбеддингов ===
EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
EMB_DIM = EMB_MODEL.get_sentence_embedding_dimension()


# === Утилиты ===
def safe_float(value, default=0.0):
    try:
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


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


# === Расчёт базовой релевантности ===
def base_similarity_score(vacancy: Dict, resume: Dict) -> Dict:
    weights = {
        "город": 0.1,
        "опыт": 0.25,
        "должность": 0.25,
        "образование": 0.15,
        "языки": 0.1,
        "зарплата": 0.1,
        "занятость": 0.05
    }

    city_match = 1.0 if vacancy["город"].lower() == resume["город"].lower() else 0.0
    exp_ratio = min(safe_float(resume["опыт"]) / safe_float(vacancy["опыт"], 1.0), 1.0)
    salary_match = 1.0 if safe_float(resume["зарплата"]) <= safe_float(vacancy["зарплата"]) else 0.8

    job_sim = cosine_sim(embed_text(vacancy["должность"]), embed_text(resume["должность"]))
    edu_sim = cosine_sim(embed_text(vacancy["образование"]), embed_text(resume["образование"]))

    langs_v = set(vacancy["языки"])
    langs_r = set(resume["языки"])
    langs_match = len(langs_v & langs_r) / len(langs_v) if langs_v else 1.0

    employment_match = 1.0 if vacancy["занятость"] == resume["занятость"] else 0.0

    score = (
        city_match * weights["город"] +
        exp_ratio * weights["опыт"] +
        job_sim * weights["должность"] +
        edu_sim * weights["образование"] +
        langs_match * weights["языки"] +
        salary_match * weights["зарплата"] +
        employment_match * weights["занятость"]
    ) * 100

    return {
        "base_score": round(score, 2),
        "details": {
            "city_match": city_match,
            "exp_ratio": exp_ratio,
            "job_sim": job_sim,
            "edu_sim": edu_sim,
            "langs_match": langs_match,
            "salary_match": salary_match,
            "employment_match": employment_match
        }
    }


# === Анализ несоответствий ===
def detect_gaps(vacancy: Dict, resume: Dict, scoring: Dict) -> List[Dict]:
    gaps = []
    d = scoring["details"]

    if vacancy["город"].lower() != resume["город"].lower():
        gaps.append({"type": "relocation", "reason": f"вакансия в {vacancy['город']}, а кандидат в {resume['город']}"})

    if d["exp_ratio"] < 0.7:
        gaps.append({"type": "training", "reason": f"опыт меньше требуемого"})

    if d["edu_sim"] < 0.6:
        gaps.append({"type": "education", "reason": "образование не полностью совпадает"})

    return gaps


# === Генерация вопросов ===
def generate_followup_question(gap, vacancy, resume, llm_client=None):
    templates = {
        "relocation": "Вакансия в {vac_city}, вы живёте в {res_city}. Готовы ли вы к переезду?",
        "training": "У вас меньше опыта, чем требуется. Готовы ли вы пройти обучение или стажировку?",
        "education": "Ваше образование немного отличается от требований. Можете пояснить, почему вы считаете себя подходящим?"
    }

    if llm_client is None:
        return templates.get(gap["type"], "Можете уточнить этот момент?").format(
            vac_city=vacancy["город"], res_city=resume["город"]
        )

    prompt = f"""
Ты HR-бот. Сформулируй короткий (до 40 слов) вопрос кандидату, чтобы уточнить следующее:
{gap['reason']}
Ответ должен начинаться с "Q:" и быть на русском языке.
"""
    return llm_client.generate(prompt)


# === Интерпретация ответов ===
def interpret_answer(answer: str) -> bool:
    answer = answer.lower()
    if any(w in answer for w in ["да", "готов", "согласен", "можно", "ok", "yes"]):
        return True
    if any(w in answer for w in ["нет", "не готов", "не могу", "не согласен"]):
        return False
    return None


# === Пересчёт итогового результата ===
def integrate_followup(vacancy, resume, scoring, followups):
    details = scoring["details"].copy()
    for f in followups:
        accepted = interpret_answer(f["answer"])
        if accepted and f["gap"]["type"] == "relocation":
            details["city_match"] = 1.0
        elif accepted and f["gap"]["type"] == "training":
            details["exp_ratio"] = min(1.0, details["exp_ratio"] + 0.2)
        elif accepted and f["gap"]["type"] == "education":
            details["edu_sim"] = min(1.0, details["edu_sim"] + 0.15)

    new_score = (
        details["city_match"] * 0.1 +
        details["exp_ratio"] * 0.25 +
        details["job_sim"] * 0.25 +
        details["edu_sim"] * 0.15 +
        details["langs_match"] * 0.1 +
        details["salary_match"] * 0.1 +
        details["employment_match"] * 0.05
    ) * 100

    return round(new_score, 2)


# === Основной сценарий ===
if __name__ == "__main__":
    vacancy = {
        "город": "Астане",
        "опыт": 5,
        "должность": "junior python developer",
        "образование": "высшее специальное",
        "языки": ["английский", "русский"],
        "зарплата": 800000,
        "занятость": "частичное"
    }

    resume = {
        "город": "Караганда",
        "опыт": 1,
        "должность": "junior python developer",
        "образование": "бакалавр компьютерных наук",
        "языки": ["русский"],
        "зарплата": 500000,
        "занятость": "полная"
    }

    llm = MistralClient()

    base = base_similarity_score(vacancy, resume)
    print(f"\n📊 Базовая релевантность: {base['base_score']}%\n")

    gaps = detect_gaps(vacancy, resume, base)
    if not gaps:
        print("✅ Несоответствий не найдено, кандидат полностью подходит.")
    else:
        print("⚠️ Обнаружены моменты, требующие уточнения:\n")

    followups = []
    for gap in gaps:
        question = generate_followup_question(gap, vacancy, resume, llm)
        print("🤖", question)
        answer = input("🧑 Ваш ответ: ").strip()
        followups.append({"gap": gap, "answer": answer})

    if followups:
        final_score = integrate_followup(vacancy, resume, base, followups)
        print(f"\n✅ Итоговая оценка после уточнений: {final_score}%")
