from PrimaryAnalysis import *

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
# print("Base:", base)

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
