# qa_logic.py
import random
from typing import Dict, List, Any
from embeddings import embed_text, cosine_sim


def generate_followup_question(gap: Dict, vacancy: Dict, resume: Dict, llm_client=None) -> str:
    """
    Генерация уточняющего вопроса по gap.
    Если есть llm_client → используем LLM, иначе — шаблоны.
    """
    templates = {
        # Переезд
        "relocation_1": "Вы указали, что живёте в {res_city}, а вакансия в {vac_city}. Рассматриваете ли вы переезд?",
        "relocation_2": "Работа в {vac_city}. Готовы ли рассмотреть переезд?",
        "relocation_3": "Если потребуется, сможете ли переехать в {vac_city}?",
        "relocation_4": "Планируете ли переезд в сторону {vac_city} в ближайшее время?",

        # experience
        "training_1": "Ваш experience ниже требуемого. Готовы ли пройти обучение/стажировку для адаптации?",
        "training_2": "Если компания предложит обучение перед стартом — согласились бы?",
        "training_3": "Готовы развить недостающие навыки через наставничество?",
        "training_4": "Готовы компенсировать недостаток experienceа за счёт практики и обучения?",

        # education
        "education_1": "Ваше education отличается от требований. Как ваш experience компенсирует это?",
        "education_2": "Есть ли курсы/практика, соответствующие требованиям вакансии?",
        "education_3": "Вы указали '{edu_res}', вакансия предполагает '{edu_vac}'. Как ваш бэкграунд помогает в роли?",
        "education_4": "Какие курсы/сертификаты закрывают требуемые компетенции?",

        # salary
        "salary_1": "По вакансии предлагается {vac_salary}₸. Устраивает ли вас этот уровень дохода?",
        "salary_2": "Ваши ожидания — {res_salary}₸, вакансия — {vac_salary}₸. Готовы рассмотреть предложение?",
        "salary_3": "Как вы оцениваете уровень {vac_salary}₸?",
        "salary_4": "Если предложат {vac_salary}₸ — будет ли это приемлемо?",

        # languages
        "language_1": "Вакансии нужны languages: {langs_vac}. Насколько уверенно вы ими владеете?",
        "language_2": "У вас указано {langs_res}, требуется {langs_vac}. Есть ли experience работы с этими языками?",
        "language_3": "Насколько комфортно общаетесь на {missing_langs}?",
        "language_4": "Готовы ли подтянуть уровень языка {missing_langs}, если потребуется?",

        # busyness
        "employment_1": "Вакансия предполагает {vac_type}, у вас {res_type}. Подходит ли формат?",
        "employment_2": "Комфортно ли вам работать в формате {vac_type}?",
        "employment_3": "Рассматриваете смену формата на {vac_type}?",
        "employment_4": "Если проект требует {vac_type} — удобно ли вам?",

        # Мотивация
        "motivation_1": "Что вас заинтересовало в вакансии?",
        "motivation_2": "Почему выбрали именно эту сферу или позицию?",
        "motivation_3": "Какие задачи вам наиболее интересны?",
        "motivation_4": "Что вас сейчас мотивирует в развитии?",
    }

    base_type = gap.get("type","")
    slot_templates = [t for k, t in templates.items() if k.startswith(base_type)]
    if slot_templates:
        t = random.choice(slot_templates)
        return t.format(
            vac_city=vacancy.get("city",""),
            res_city=resume.get("city",""),
            vac_salary=vacancy.get("salary",""),
            res_salary=resume.get("salary",""),
            langs_vac=", ".join(vacancy.get("languages",[])) or "требуемые languages",
            langs_res=", ".join(resume.get("languages",[])) or "указанные languages",
            missing_langs=gap.get("missing_langs","указанные languages"),
            vac_type=vacancy.get("busyness",""),
            res_type=resume.get("busyness",""),
            edu_vac=vacancy.get("education",""),
            edu_res=resume.get("education",""),
        )

    if llm_client:
        prompt = (
            "Сформулируй короткий (<=40 слов) вежливый вопрос кандидату,"
            f" чтобы уточнить: {gap.get('reason','несоответствие')}. "
            'Ответ начни c "Q:".'
        )
        return llm_client.generate(prompt)

    return "Можете уточнить этот момент подробнее?"


def interpret_answer(answer: str) -> bool:
    """
    Быстрая интерпретация: сначала ключевые слова, затем эмбеддинги.
    Возвращает True | False | None.
    """
    text = (answer or "").lower().strip()
    if any(w in text for w in ["да", "готов", "соглас", "рассмотр", "возмож", "не против"]):
        return True
    if any(w in text for w in ["нет", "не готов", "не хочу", "не могу", "невозможно", "не уверен"]):
        return False

    # fallback: по смыслу через эмбеддинги
    positives = ["да", "готов", "готов переехать", "готов учиться", "согласен", "без проблем"]
    negatives = ["нет", "не готов", "не хочу", "не могу", "это невозможно", "я не уверен"]
    ans_emb = embed_text(answer)
    pos = sum(cosine_sim(ans_emb, embed_text(p)) for p in positives) / max(1, len(positives))
    neg = sum(cosine_sim(ans_emb, embed_text(n)) for n in negatives) / max(1, len(negatives))
    if pos > neg + 0.08:
        return True
    if neg > pos + 0.08:
        return False
    return None


def integrate_followup(vacancy: Dict, resume: Dict, scoring: Dict, followups: List[Dict]) -> float:
    """
    Применяем ответы кандидата к деталям скоринга и возвращаем новый итоговый score.
    """
    details = scoring["details"].copy()
    updated = False

    for f in followups:
        accepted = interpret_answer(f.get("answer",""))
        if not accepted:
            continue
        updated = True
        t = f["gap"]["type"]
        if t == "relocation":
            details["city_match"] = 1.0
        elif t == "training":
            details["exp_ratio"] = min(1.0, details["exp_ratio"] + 0.3)
        elif t == "education":
            details["edu_sim"] = min(1.0, details["edu_sim"] + 0.25)
        elif t == "language":
            details["langs_match"] = min(1.0, details["langs_match"] + 0.3)
        elif t == "salary":
            details["salary_match"] = min(1.0, details["salary_match"] + 0.25)
        elif t == "employment":
            details["employment_match"] = 1.0

    if not updated:
        return scoring["base_score"]

    new_score = (
        details["city_match"] * 0.1 +
        details["exp_ratio"] * 0.25 +
        details["job_sim"] * 0.25 +
        details["edu_sim"] * 0.15 +
        details["langs_match"] * 0.1 +
        details["salary_match"] * 0.1 +
        details["employment_match"] * 0.05
    ) * 100.0
    return round(new_score, 2)
