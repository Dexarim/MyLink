# main.py (опционально — для ручных проверок в консоли)
import os
from llm_client import LLMClient
from scoring import base_similarity_score, detect_gaps
from qa_logic import generate_followup_question, integrate_followup

if __name__ == "__main__":
    vacancy = {
        "city": "Москва",
        "experience": 2,
        "post": "Frontend-разработчик",
        "education": "Бакалавр компьютерных наук",
        "languages": ["русский", "английский"],
        "salary": 700000,
        "busyness": "полная",
          "skills": [
            "React",
            "JavaScript",
            "TypeScript",
            "REST API"
        ],
    }

    resume = {
        "city": "Казань",
        "experience": 7,
        "post": "Сварщик",
        "education": "Среднее профессиональное, Казанский техникум машиностроения",
        "languages": ["русский"],
        "salary": 450000,
        "busyness": "полная",
        "skills": [
    "Электросварка",
    "Аргонная сварка",
    "Чтение чертежей",
    "Металлоконструкции",
    "Техническое обслуживание оборудования"
    ],
  "motivation": "Люблю работать руками, ценю качество и точность. Хочу развиваться в промышленной сварке и участвовать в крупных проектах."
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
        print("", q)
        ans = input("Ваш ответ: ").strip()
        followups.append({"gap": g, "answer": ans})

    if followups:
        final = integrate_followup(vacancy, resume, base, followups)
        print("Итоговая оценка:", final)
