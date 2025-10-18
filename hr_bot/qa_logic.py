import numpy as np
from embeddings import embed_text, cosine_sim



# === Генерация вопросов для уточнения ===
def generate_followup_question(gap, vacancy, resume, llm_client=None):
    """
    Формирует естественный уточняющий вопрос кандидату в зависимости от обнаруженного расхождения (gap).
    Если есть llm_client, использует его для генерации индивидуального вопроса.
    """

    templates = {
        # === Переезд / город ===
        "relocation_1": "Вы указали, что живёте в {res_city}, а вакансия находится в {vac_city}. Рассматриваете ли вы возможность переезда?",
        "relocation_2": "Работа предлагается в {vac_city}. Готовы ли вы рассмотреть переезд в этот город?",
        "relocation_3": "Если потребуется, сможете ли вы переехать в {vac_city} для работы по данной вакансии?",
        "relocation_4": "Планируете ли вы в ближайшее время смену места жительства, например в сторону {vac_city}?",

        # === Недостаток опыта ===
        "training_1": "Ваш опыт немного меньше, чем требуется. Готовы ли вы пройти обучение или стажировку для адаптации?",
        "training_2": "Если компания предложит обучение перед началом работы — согласились бы вы?",
        "training_3": "Хотели бы вы развить недостающие навыки через корпоративное обучение или наставничество?",
        "training_4": "Готовы ли вы компенсировать недостаток опыта за счёт практики и обучения?",

        # === Образование ===
        "education_1": "Ваше образование отличается от требований вакансии. Расскажите, как ваш опыт помогает компенсировать это различие?",
        "education_2": "Хотя профиль вашего образования другой, есть ли у вас практические знания, подходящие под требования?",
        "education_3": "Вы указали образование '{edu_res}', а вакансия предполагает '{edu_vac}'. Как ваш бэкграунд помогает в этой роли?",
        "education_4": "Какие курсы или опыт помогли вам освоить компетенции, требуемые для этой вакансии?",

        # === Зарплатные ожидания ===
        "salary_1": "По вакансии предлагается {vac_salary}₸. Устраивает ли вас такой уровень дохода?",
        "salary_2": "Ваши ожидания — {res_salary}₸, а по вакансии — {vac_salary}₸. Готовы рассмотреть данное предложение?",

        "salary_3": "Как вы оцениваете уровень дохода в {vac_salary}₸, предлагаемый по вакансии?",
        "salary_4": "Если компания предложит {vac_salary}₸, будет ли это приемлемо для вас?",

        # === Владение языками ===
        "language_1": "В вакансии требуется знание языков: {langs_vac}. Насколько уверенно вы ими владеете?",
        "language_2": "Вы указали владение {langs_res}, а вакансия требует {langs_vac}. Есть ли у вас опыт работы с этими языками?",
        "language_3": "Насколько комфортно вы чувствуете себя при общении или работе на {missing_langs}?",
        "language_4": "Если потребуется, готовы ли вы подтянуть уровень владения языком {missing_langs}?",

        # === Тип занятости ===
        "employment_1": "Вакансия предполагает {vac_type} занятость, а вы указали {res_type}. Подходит ли вам такой формат работы?",
        "employment_2": "Комфортно ли вам работать в формате {vac_type}?",
        "employment_3": "Рассматриваете ли вы возможность сменить формат работы на {vac_type}?",
        "employment_4": "Если проект будет требовать {vac_type} графика, это будет удобно для вас?",

        # === Мотивация / интерес к вакансии ===
        "motivation_1": "Что вас заинтересовало в этой вакансии?",
        "motivation_2": "Почему вы выбрали именно эту сферу или позицию?",
        "motivation_3": "Какие задачи или направления работы для вас наиболее привлекательны?",
        "motivation_4": "Что мотивирует вас в профессиональном развитии сейчас?",

        # === Общие уточнения ===
        "general_1": "Расскажите, какие ваши сильные стороны помогут успешно справиться с этой работой?",
        "general_2": "Есть ли у вас опыт, который может быть полезен для этой должности, но не отражён в резюме?",
        "general_3": "Что вы ожидаете от будущей работы в первую очередь?",
        "general_4": "Какие факторы для вас наиболее важны при выборе новой позиции?",
    }

    # === Если есть готовый шаблон для gap ===
    base_type = gap.get("type", "")
    available_templates = [v for k, v in templates.items() if k.startswith(base_type)]

    if available_templates:
        import random
        template = random.choice(available_templates)
        return template.format(
            vac_city=vacancy.get("город", "указанном городе"),
            res_city=resume.get("город", "вашем городе"),
            vac_salary=vacancy.get("зарплата", "указанную сумму"),
            res_salary=resume.get("зарплата", "ваши ожидания"),
            langs_vac=vacancy.get("языки", "требуемые языки"),
            langs_res=resume.get("языки", "указанные вами языки"),
            missing_langs=gap.get("missing_langs", "указанные языки"),
            vac_type=vacancy.get("занятость", "полную"),
            res_type=resume.get("занятость", "вашу"),
            edu_vac=vacancy.get("образование", "требуемое образование"),
            edu_res=resume.get("образование", "ваше образование"),
        )

    # === Если шаблона нет — используем LLM для генерации ===
    if llm_client:
        prompt = f"""
        Ты выступаешь как HR-бот. Сформулируй короткий (до 40 слов) вопрос кандидату, чтобы уточнить:
        {gap.get('reason', 'неясный момент между резюме и вакансией')}.
        Вопрос должен быть дружелюбным, профессиональным и начинаться с "Q:".
        """
        response = llm_client.generate(prompt)
        return response.strip()

    # === fallback на случай отсутствия llm и шаблона ===
    return "Можете уточнить этот момент подробнее?"


# === Улучшенная интерпретация ответов через эмбеддинги ===
def interpret_answer(answer: str) -> bool:
    """
    Интерпретация ответа: сначала простое правило, потом — эмбеддинги.
    """

    positive_samples = [
        "да", "конечно", "готов", "без проблем", "можно", "согласен", "буду рад", "не против", "да, конечно",
        "готов пройти обучение", "готов переехать", "готов учиться"
    ]
    negative_samples = [
        "нет", "не могу", "не готов", "не хочу", "не согласен", "нет возможности", "это невозможно", "я не уверен"
    ]
    text = answer.lower().strip()

    # === Быстрая проверка по словам ===
    if any(word in text for word in ["да", "готов", "соглас", "рассмотр", "возмож", "не против"]):
        return True
    if any(word in text for word in ["нет", "не готов", "не хочу", "не могу", "невозможно", "не уверен"]):
        return False

    # === Если не ясно — fallback на эмбеддинги ===
    ans_emb = embed_text(answer)
    pos_embs = [embed_text(p) for p in positive_samples]
    neg_embs = [embed_text(n) for n in negative_samples]

    pos_score = np.mean([cosine_sim(ans_emb, e) for e in pos_embs])
    neg_score = np.mean([cosine_sim(ans_emb, e) for e in neg_embs])

    if pos_score > neg_score + 0.1:
        return True
    elif neg_score > pos_score + 0.1:
        return False
    else:
        return None


# === Пересчёт итогового результата ===
def integrate_followup(vacancy, resume, scoring, followups):
    details = scoring["details"].copy()
    updated = False  # флаг, что что-то реально изменилось

    for f in followups:
        accepted = interpret_answer(f["answer"])
        if accepted:
            updated = True
            t = f["gap"]["type"]
            if t == "relocation":
                details["city_match"] = 1.0
            elif t == "training":
                details["exp_ratio"] = min(1.0, details["exp_ratio"] + 0.3)
            elif t == "education":
                details["edu_sim"] = min(1.0, details["edu_sim"] + 0.25)

    if not updated:
        print("⚠️ Ответы не были интерпретированы как положительные, результат не изменён.")
        return scoring["base_score"]

    new_score = (
        details["city_match"] * 0.1 +
        details["exp_ratio"] * 0.25 +
        details["job_sim"] * 0.25 +
        details["edu_sim"] * 0.15 +
        details["langs_match"] * 0.1 +
        details["salary_match"] * 0.1 +
        details["employment_match"] * 0.05
    ) * 100

    diff = new_score - scoring["base_score"]
    print(f"⬆️ Улучшение после ответов: +{diff:.2f} пунктов")

    return round(new_score, 2)