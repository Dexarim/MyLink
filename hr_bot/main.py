from llm_client import MistralClient
from scoring import base_similarity_score, detect_gaps
from qa_logic import generate_followup_question, integrate_followup

vacancy = {
        "город": "Астанa",
        "опыт": 5,
        "должность": "junior python developer",
        "образование": "высшее специальное",
        "языки": ["английский"],
        "зарплата": 800000,
        "занятость": "частичное"
    }

resume = {
        "город": "Астанa",
        "опыт": 1,
        "должность": "junior python developer",
        "образование": "бакалавр компьютерных наук",
        "языки": ["русский"],
        "зарплата": 500000,
        "занятость": "полная"
    }

if __name__ == "__main__":
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
