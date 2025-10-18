# main.py (опционально — для ручных проверок в консоли)
import os
from llm_client import LLMClient
from scoring import base_similarity_score, detect_gaps
from qa_logic import generate_followup_question, integrate_followup

if __name__ == "__main__":
    vacancy = {
        "город": "Астана",
        "опыт": 5,
        "должность": "Python developer",
        "образование": "высшее специальное",
        "языки": ["английский"],
        "зарплата": 800000,
        "занятость": "полная",
        "skills": ["python", "fastapi", "numpy"]
    }

    resume = {
        "город": "Караганда",
        "опыт": 1,
        "должность": "junior python developer",
        "образование": "бакалавр компьютерных наук",
        "языки": ["русский"],
        "зарплата": 500000,
        "занятость": "полная",
        "skills": ["python", "django", "sql"],
        "мотивация": "Хочу развиваться в Python backend"
    }

    llm = LLMClient(provider=os.getenv("LLM_PROVIDER","gemini"),
                    model=os.getenv("LLM_MODEL","gemini-1.5-flash"),
                    api_key=os.getenv("GOOGLE_API_KEY"))

    base = base_similarity_score(vacancy, resume)
    print("Base:", base)

    gaps = detect_gaps(vacancy, resume, base)
    print("Gaps:", gaps)

    followups = []
    for g in gaps:
        q = generate_followup_question(g, vacancy, resume, llm)
        print("🤖", q)
        ans = input("🧑 Ваш ответ: ").strip()
        followups.append({"gap": g, "answer": ans})

    if followups:
        final = integrate_followup(vacancy, resume, base, followups)
        print("✅ Итоговая оценка:", final)
