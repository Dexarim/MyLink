from typing import Dict
from embeddings import cosine_sim, embed_text, safe_float


def base_similarity_score(vacancy: Dict, resume: Dict) -> Dict:
    """
    Базовый скоринг (0..100) по основным полям.
    Добавлен анализ схожести навыков (skills) и штраф за слабое совпадение профессий (post).
    """
    weights = {
        "city": 0.1,
        "experience": 0.25,
        "post": 0.2,
        "education": 0.15,
        "languages": 0.1,
        "salary": 0.1,
        "busyness": 0.05,
        "skills": 0.05  # новый параметр
    }

    # ---------- Совпадение города ----------
    city_match = 1.0 if vacancy.get("city", "").lower() == resume.get("city", "").lower() else 0.0

    # ---------- Опыт работы ----------
    exp_ratio = 1.0
    vac_exp = safe_float(vacancy.get("experience"), 0.0)
    res_exp = safe_float(resume.get("experience"), 0.0)
    if vac_exp > 0:
        exp_ratio = min(res_exp / vac_exp, 1.0)

    # ---------- Зарплата ----------
    salary_match = 1.0
    res_salary = safe_float(resume.get("salary"), 0.0)
    vac_salary = safe_float(vacancy.get("salary"), 0.0)
    if res_salary and vac_salary:
        salary_match = 1.0 if res_salary <= vac_salary else max(0.0, vac_salary / res_salary)

    # ---------- Должность и образование ----------
    job_sim = cosine_sim(embed_text(vacancy.get("post", "")), embed_text(resume.get("post", "")))
    edu_sim = cosine_sim(embed_text(vacancy.get("education", "")), embed_text(resume.get("education", "")))

    # ---------- Языки ----------
    vac_langs = set([l.lower() for l in vacancy.get("languages", [])])
    res_langs = set([l.lower() for l in resume.get("languages", [])])
    langs_match = len(vac_langs & res_langs) / len(vac_langs) if vac_langs else 1.0

    # ---------- Тип занятости ----------
    employment_match = 1.0 if vacancy.get("busyness") == resume.get("busyness") else 0.0

    # ---------- Навыки ----------
    vac_skills = " ".join(vacancy.get("skills", []))
    res_skills = " ".join(resume.get("skills", []))
    skills_sim = cosine_sim(embed_text(vac_skills), embed_text(res_skills))

    # ---------- Общий скоринг ----------
    score = (
        city_match * weights["city"] +
        exp_ratio * weights["experience"] +
        job_sim * weights["post"] +
        edu_sim * weights["education"] +
        langs_match * weights["languages"] +
        salary_match * weights["salary"] +
        employment_match * weights["busyness"] +
        skills_sim * weights["skills"]
    ) * 100.0

    # ---------- Штраф за низкое совпадение профессий ----------
    if job_sim < 0.4:
        score *= 0.6  # уменьшаем итоговый балл на 40%

    return {
        "base_score": round(score, 2),
        "details": {
            "city_match": round(city_match, 3),
            "exp_ratio": round(exp_ratio, 3),
            "job_sim": round(job_sim, 3),
            "edu_sim": round(edu_sim, 3),
            "langs_match": round(langs_match, 3),
            "salary_match": round(salary_match, 3),
            "employment_match": round(employment_match, 3),
            "skills_sim": round(skills_sim, 3)
        }
    }


def detect_gaps(vacancy: Dict, resume: Dict, scoring: Dict):
    """
    Выявление несоответствий по результатам base_similarity_score.
    Добавлено предупреждение при низком совпадении профессии (post).
    """
    gaps = []
    d = scoring["details"]

    # ---------- Город ----------
    if vacancy.get("city") and resume.get("city") and vacancy["city"].lower() != resume["city"].lower():
        gaps.append({"type": "relocation", "reason": f"вакансия в {vacancy['city']}, кандидат в {resume['city']}"})

    # ---------- Опыт ----------
    if d["exp_ratio"] < 0.7:
        gaps.append({"type": "training", "reason": "experience меньше требуемого"})

    # ---------- Образование ----------
    if d["edu_sim"] < 0.6:
        gaps.append({"type": "education", "reason": "education не полностью совпадает"})

    # ---------- Зарплата ----------
    res_salary = safe_float(resume.get("salary"), 0)
    vac_salary = safe_float(vacancy.get("salary"), 0)
    if res_salary > 0 and vac_salary > 0 and res_salary > vac_salary * 1.15:
        gaps.append({"type": "salary", "reason": f"ожидания {res_salary} > оффер {vac_salary}"})

    # ---------- Языки ----------
    req_langs = set([l.lower() for l in vacancy.get("languages", [])])
    res_langs = set([l.lower() for l in resume.get("languages", [])])
    missing = list(req_langs - res_langs)
    if missing:
        gaps.append({
            "type": "language",
            "reason": f"отсутствуют требуемые languages: {', '.join(missing)}",
            "missing_langs": ", ".join(missing)
        })

    # ---------- Тип занятости ----------
    if vacancy.get("busyness") and resume.get("busyness") and vacancy["busyness"] != resume["busyness"]:
        gaps.append({"type": "employment", "reason": "тип занятости не совпадает"})

    # ---------- Мотивация ----------
    if not resume.get("мотивация") or len(resume.get("мотивация", "")) < 30:
        gaps.append({"type": "motivation", "reason": "мало информации о мотивации"})

    # ---------- Новое: низкое совпадение профессии ----------
    if d["job_sim"] < 0.4:
        gaps.append({
            "type": "profession_mismatch",
            "reason": "низкое совпадение профессии — кандидат работает в другой сфере"
        })

    # ---------- Новое: низкое совпадение навыков ----------
    if d["skills_sim"] < 0.5:
        gaps.append({
            "type": "skills_gap",
            "reason": "навыки кандидата сильно отличаются от требований вакансии"
        })

    return gaps
