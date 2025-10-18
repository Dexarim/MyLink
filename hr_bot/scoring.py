# scoring.py
from typing import Dict
from embeddings import cosine_sim, embed_text, safe_float


def base_similarity_score(vacancy: Dict, resume: Dict) -> Dict:
    """
    Базовый скоринг (0..100) по основным полям.
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

    city_match = 1.0 if vacancy.get("город","").lower() == resume.get("город","").lower() else 0.0

    exp_ratio = 1.0
    vac_exp = safe_float(vacancy.get("опыт"), 0.0)
    res_exp = safe_float(resume.get("опыт"), 0.0)
    if vac_exp > 0:
        exp_ratio = min(res_exp / vac_exp, 1.0)

    salary_match = 1.0
    res_salary = safe_float(resume.get("зарплата"), 0.0)
    vac_salary = safe_float(vacancy.get("зарплата"), 0.0)
    if res_salary and vac_salary:
        salary_match = 1.0 if res_salary <= vac_salary else max(0.0, vac_salary / res_salary)

    job_sim = cosine_sim(embed_text(vacancy.get("должность","")), embed_text(resume.get("должность","")))
    edu_sim = cosine_sim(embed_text(vacancy.get("образование","")), embed_text(resume.get("образование","")))

    vac_langs = set([l.lower() for l in vacancy.get("языки", [])])
    res_langs = set([l.lower() for l in resume.get("языки", [])])
    langs_match = len(vac_langs & res_langs) / len(vac_langs) if vac_langs else 1.0

    employment_match = 1.0 if vacancy.get("занятость") == resume.get("занятость") else 0.0

    score = (
        city_match * weights["город"] +
        exp_ratio * weights["опыт"] +
        job_sim * weights["должность"] +
        edu_sim * weights["образование"] +
        langs_match * weights["языки"] +
        salary_match * weights["зарплата"] +
        employment_match * weights["занятость"]
    ) * 100.0

    return {
        "base_score": round(score, 2),
        "details": {
            "city_match": round(city_match, 3),
            "exp_ratio": round(exp_ratio, 3),
            "job_sim": round(job_sim, 3),
            "edu_sim": round(edu_sim, 3),
            "langs_match": round(langs_match, 3),
            "salary_match": round(salary_match, 3),
            "employment_match": round(employment_match, 3)
        }
    }


def detect_gaps(vacancy: Dict, resume: Dict, scoring: Dict):
    """
    Выявление несоответствий по результатам base_similarity_score.
    """
    gaps = []
    d = scoring["details"]

    # город
    if vacancy.get("город") and resume.get("город") and vacancy["город"].lower() != resume["город"].lower():
        gaps.append({"type": "relocation", "reason": f"вакансия в {vacancy['город']}, кандидат в {resume['город']}"})

    # опыт
    if d["exp_ratio"] < 0.7:
        gaps.append({"type": "training", "reason": "опыт меньше требуемого"})

    # образование
    if d["edu_sim"] < 0.6:
        gaps.append({"type": "education", "reason": "образование не полностью совпадает"})

    # зарплата
    res_salary = safe_float(resume.get("зарплата"), 0)
    vac_salary = safe_float(vacancy.get("зарплата"), 0)
    if res_salary > 0 and vac_salary > 0 and res_salary > vac_salary * 1.15:
        gaps.append({"type": "salary", "reason": f"ожидания {res_salary} > оффер {vac_salary}"})

    # языки
    req_langs = set([l.lower() for l in vacancy.get("языки", [])])
    res_langs = set([l.lower() for l in resume.get("языки", [])])
    missing = list(req_langs - res_langs)
    if missing:
        gaps.append({"type": "language", "reason": f"отсутствуют требуемые языки: {', '.join(missing)}", "missing_langs": ", ".join(missing)})

    # занятость
    if vacancy.get("занятость") and resume.get("занятость") and vacancy["занятость"] != resume["занятость"]:
        gaps.append({"type": "employment", "reason": "тип занятости не совпадает"})

    # мотивация (по желанию)
    if not resume.get("мотивация") or len(resume.get("мотивация","")) < 30:
        gaps.append({"type": "motivation", "reason": "мало информации о мотивации"})

    return gaps
