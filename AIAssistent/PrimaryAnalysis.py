"""
Прототип системы релевантности с follow-up через LLM (Mistral)
Требования:
- python 3.8
- sentence-transformers
- transformers (если используешь локальный mistral)
- rapidfuzz (опционально)
- requests / huggingface_hub / openai (в зависимости от варианта запуска LLM)

Заменяй ID моделей и ключи на свои.
"""

import re
import json
from typing import Dict, List, Tuple, Any

# Для эмбеддингов
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Для локального/инференс-запроса к Mistral:
# Вариант A: transformers (локально, если модель доступна)
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Вариант B: вызов внешнего inference API (Hugging Face / OpenAI) — оставлю шаблон ниже.

# === Инициализация эмбеддингов (один раз при старте) ===
EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")  # лёгкая и быстрая модель


# === Утилиты ===
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def embed_text(text: str) -> np.ndarray:
    return EMB_MODEL.encode(text, convert_to_numpy=True)


# === Начальная функция оценки (без follow-up) ===
def base_similarity_score(vacancy: Dict, resume: Dict, embeddings_cache: Dict = None) -> Dict:
    """
    Возвращает dict с:
      - base_score (0..100)
      - компоненты с деталями (job_sim, edu_sim, city_match, exp_ratio и т.д.)
    """
    weights = {
        "город": 0.1,
        "опыт": 0.25,
        "должность": 0.25,
        "образование": 0.15,
        "языки": 0.1,
        "зарплата": 0.1,
        "занятость": 0.05
    }

    # город
    city_match = 1.0 if vacancy.get("город", "").strip().lower() == resume.get("город", "").strip().lower() else 0.0

    # опыт
    vac_exp = float(vacancy.get("опыт", 0))
    res_exp = float(resume.get("опыт", 0))
    exp_ratio = min(res_exp / max(1e-6, vac_exp), 1.0) if vac_exp > 0 else 1.0

    # зарплата
    res_salary = float(resume.get("зарплата", 0))
    vac_salary = float(vacancy.get("зарплата", 0))
    salary_match = 1.0 if res_salary <= vac_salary else max(0.0, vac_salary / res_salary)  # простая логика

    # языки
    vac_langs = [l.lower() for l in vacancy.get("языки", [])]
    res_langs = [l.lower() for l in resume.get("языки", [])]
    langs_match = len(set(vac_langs) & set(res_langs)) / max(1, len(vac_langs))

    # занятость
    employment_match = 1.0 if vacancy.get("занятость") == resume.get("занятость") else 0.0

    # должность и образование — через эмбеддинги
    # кэширование эмбеддингов (чтобы не считать повторно)
    if embeddings_cache is None:
        embeddings_cache = {}
    def get_emb(key, text):
        if text is None:
            return np.zeros(384)
        if key not in embeddings_cache:
            embeddings_cache[key] = embed_text(text)
        return embeddings_cache[key]

    job_emb_v = get_emb("vac_job:" + vacancy.get("должность",""), vacancy.get("должность",""))
    job_emb_r = get_emb("res_job:" + resume.get("должность",""), resume.get("должность",""))
    job_sim = cosine_sim(job_emb_v, job_emb_r)

    edu_emb_v = get_emb("vac_edu:" + vacancy.get("образование",""), vacancy.get("образование",""))
    edu_emb_r = get_emb("res_edu:" + resume.get("образование",""), resume.get("образование",""))
    edu_sim = cosine_sim(edu_emb_v, edu_emb_r)

    # итоговый score (0..100)
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
            "exp_ratio": round(exp_ratio, 2),
            "job_sim": round(job_sim, 3),
            "edu_sim": round(edu_sim, 3),
            "langs_match": round(langs_match, 3),
            "salary_match": round(salary_match, 3),
            "employment_match": employment_match
        },
        "embeddings_cache": embeddings_cache
    }


# === Детектор gaps / триггеров для follow-up ===
def detect_gaps(vacancy: Dict, resume: Dict, scoring: Dict) -> List[Dict]:
    """
    Возвращает список задач/вопросов, которые нужно задать кандидату.
    Каждый элемент: {"type": "relocation"|"training"|"salary_negotiation"|..., "reason": "..."}
    """
    gaps = []
    details = scoring["details"]

    # 1) город != вакансия -> relocation вопрос (если вакансии не remote)
    if vacancy.get("город") and resume.get("город") and vacancy.get("город").strip().lower() != resume.get("город").strip().lower():
        # если вакансия remote — пропускаем
        if vacancy.get("занятость") != "удаленно" and not vacancy.get("remote_allowed", False):
            gaps.append({"type": "relocation", "reason": f"вакансия в {vacancy.get('город')}, кандидат в {resume.get('город')}"})

    # 2) опыт значительно меньше требуемого (например, менее 70%)
    if details["exp_ratio"] < 0.7:
        gaps.append({"type": "training", "reason": f"опыт {resume.get('опыт')} лет при требуемых {vacancy.get('опыт')}"})

    # 3) зарплатные ожидания выше вакансии более чем на 15%
    res_sal = float(resume.get("зарплата", 0))
    vac_sal = float(vacancy.get("зарплата", 0))
    if res_sal > 0 and vac_sal > 0 and res_sal > vac_sal * 1.15:
        gaps.append({"type": "salary", "reason": f"ожидания {res_sal} > оффер {vac_sal}"})

    # 4) языки: если вакансии нужен язык, которого нет в резюме
    req_langs = set([l.lower() for l in vacancy.get("языки", [])])
    res_langs = set([l.lower() for l in resume.get("языки", [])])
    missing = list(req_langs - res_langs)
    if missing:
        gaps.append({"type": "languages", "reason": f"отсутствует язык(и): {', '.join(missing)}"})

    return gaps


# === Генерация вопроса через LLM (Mistral) ===
def generate_followup_question(gap: Dict, vacancy: Dict, resume: Dict, llm_client: Any = None) -> str:
    """
    Возвращает формулировку вопроса, которую нужно задать кандидату.
    llm_client — объект/функция для вызова Mistral; можно подставить свой wrapper.
    Если llm_client отсутствует, используем шаблоны.
    """
    typ = gap["type"]
    # шаблоны (простые)
    templates = {
        "relocation": (
            "Вакансия расположена в {vac_city}, а в вашем резюме указан город {res_city}. "
            "Готовы ли вы переехать в {vac_city} или рассматриваете удалённую работу?"
        ),
        "training": (
            "В описании вакансии требуется ~{vac_exp} лет опыта, у вас указано {res_exp} лет. "
            "Готовы ли вы пройти обучение/менторство и работать в роли с частичной поддержкой?"
        ),
        "salary": (
            "Ваши зарплатные ожидания — {res_sal}, а в вакансии указано {vac_sal}. Готовы ли вы обсуждать зарплату?"
        ),
        "languages": (
            "В вакансии требуется знание {req_langs}. У вас указаны: {res_langs}. Можете ли вы подтянуть/подтвердить знание {req_langs}?"
        )
    }

    if llm_client is None:
        # подставляем переменные
        return templates.get(typ, "Можем уточнить: вы готовы к этому?").format(
            vac_city=vacancy.get("город","не указан"),
            res_city=resume.get("город","не указан"),
            vac_exp=vacancy.get("опыт","?"),
            res_exp=resume.get("опыт","?"),
            res_sal=resume.get("зарплата","?"),
            vac_sal=vacancy.get("зарплата","?"),
            req_langs=", ".join(vacancy.get("языки", [])),
            res_langs=", ".join(resume.get("языки", []))
        )

    # Если есть клиент LLM — мы можем сгенерировать более мягкий, персонализованный вопрос:
    prompt = f"""Сгенерируй краткий вежливый вопрос к кандидату, на русском языке, не более 40 слов.
    Повод: {gap['reason']}.
    Вакансия: {vacancy.get('должность')} в {vacancy.get('город')}.
    Резюме: {resume.get('должность')}, город {resume.get('город')}, опыт {resume.get('опыт')} лет.
    """

    # вызов llm_client должен вернуть строку; пример вызова зависит от API/клиента, который используешь.
    return llm_client.generate(prompt)


# === Нормализация ответа кандидата (используем LLM кратко) ===
def interpret_answer(answer_text: str, question_type: str, llm_client: Any = None) -> Dict:
    """
    Возвращает структуру с полями, которые помогут скору:
     - accepted: True/False/partial
     - confidence: 0..1 (оценка LLM или heuristics)
     - raw: оригинальный ответ
    Если нет llm_client — используем простые правила.
    """
    text = answer_text.strip().lower()
    if llm_client is None:
        # простая эвристика
        yes_words = ["да", "готов", "готовы", "ok", "okey", "yes", "можно"]
        no_words = ["нет", "не готов", "неготов", "не рассматриваю"]
        if any(w in text for w in yes_words):
            return {"accepted": True, "confidence": 0.9, "raw": answer_text}
        if any(w in text for w in no_words):
            return {"accepted": False, "confidence": 0.9, "raw": answer_text}
        # иначе частичный
        return {"accepted": "partial", "confidence": 0.5, "raw": answer_text}

    # c LLM можно попросить классифицировать ответ и вернуть JSON
    prompt = f"""Классифицируй кратко ответ кандидата на русский вопрос по типу "{question_type}".
    Верни JSON: {{ "accepted": true/false/"partial", "confidence": 0..1, "notes": "..." }}
    Ответ кандидата: \"\"\"{answer_text}\"\"\"
    """
    parsed = llm_client.generate_json(prompt)
    return parsed


# === Интеграция ответов в скор ===
def integrate_followup_and_recompute(vacancy: Dict, resume: Dict, base_scoring: Dict, followups: List[Tuple[Dict, str]], embeddings_cache: Dict) -> Dict:
    """
    followups: список (gap, answer_text)
    возвращает обновлённый score и объяснения
    """
    # копируем детали
    details = base_scoring["details"].copy()
    extra_bonus = 0.0

    for gap, answer in followups:
        interp = interpret_answer(answer, gap["type"], llm_client=None)  # можно поставить llm_client
        if gap["type"] == "relocation":
            if interp["accepted"] is True:
                # если согласен переехать — даём бонус к city_match
                details["city_match"] = 1.0
                extra_bonus += 0.02  # небольшой бонус за готовность
        if gap["type"] == "training":
            if interp["accepted"] is True:
                # кандидат готов учиться — считаем опыт как более высокий (например +20% от недостающего)
                details["exp_ratio"] = min(1.0, details["exp_ratio"] + 0.2 * interp["confidence"])
        if gap["type"] == "salary":
            if interp["accepted"] is True:
                # готов обсуждать — улучшаем salary_match
                details["salary_match"] = min(1.0, details.get("salary_match", 0.0) + 0.25 * interp["confidence"])
        if gap["type"] == "languages":
            if interp["accepted"] is True:
                # кандидат сказал, что подтянет язык — даём частичную поправку
                details["langs_match"] = min(1.0, details.get("langs_match", 0.0) + 0.5 * interp["confidence"])

    # пересчёт итогового скор, учитывая веса те же что и base
    weights = {
        "город": 0.1,
        "опыт": 0.25,
        "должность": 0.25,
        "образование": 0.15,
        "языки": 0.1,
        "зарплата": 0.1,
        "занятость": 0.05
    }
    # берем job_sim и edu_sim из base_scoring (они не менялись)
    job_sim = base_scoring["details"]["job_sim"]
    edu_sim = base_scoring["details"]["edu_sim"]

    new_score = (
        details["city_match"] * weights["город"] +
        details["exp_ratio"] * weights["опыт"] +
        job_sim * weights["должность"] +
        edu_sim * weights["образование"] +
        details["langs_match"] * weights["языки"] +
        details.get("salary_match", base_scoring["details"].get("salary_match", 1.0)) * weights["зарплата"] +
        details["employment_match"] * weights["занятость"]
    ) * 100 + extra_bonus * 100

    return {
        "updated_score": round(new_score, 2),
        "updated_details": details
    }


# === Пример workflow ===
if __name__ == "__main__":
    vacancy = {
        "город": "Алматы",
        "опыт": 3,
        "должность": "Python разработчик",
        "образование": "высшее техническое",
        "языки": ["английский", "русский"],
        "зарплата": 600000,
        "занятость": "полная",
        "remote_allowed": False
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

    base = base_similarity_score(vacancy, resume)
    print("Base:", base)

    gaps = detect_gaps(vacancy, resume, base)
    print("Gaps to ask:", gaps)

    # генерируем вопросы (без LLM, по шаблону)
    questions = [generate_followup_question(g, vacancy, resume) for g in gaps]
    for q in questions:
        print("Q:", q)

    # <-- здесь в реальной системе мы отправляем вопросы пользователю и получаем ответы -->
    # для примера задаём ответы вручную:
    answers = [
        ("relocation", "Да, готов переехать через 2 месяца"),
        ("training", "Да, готов учиться и пройти стажировку")
    ]

    # мапим ответы на gaps
    followups = []
    for g in gaps:
        # находим ответ по типу (в примере)
        ans = next((a for a in answers if a[0] == g["type"]), ("", ""))[1]
        followups.append((g, ans))

    updated = integrate_followup_and_recompute(vacancy, resume, base, followups, base["embeddings_cache"])
    print("Updated:", updated)
