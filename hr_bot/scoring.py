# scoring.py
from typing import Dict
from embeddings import cosine_sim, embed_text, safe_float


def base_similarity_score(vacancy: Dict, resume: Dict) -> Dict:
    """
    Базовый скоринг (0..100) по основным полям.
    """
    weights = {
        "city": 0.1,
        "experience": 0.25,
        "post": 0.25,
        "education": 0.15,
        "languages": 0.1,
        "salary": 0.1,
        "busyness": 0.05
    }

    city_match = 1.0 if vacancy.get("city","").lower() == resume.get("city","").lower() else 0.0

    exp_ratio = 1.0
    vac_exp = safe_float(vacancy.get("experience"), 0.0)
    res_exp = safe_float(resume.get("experience"), 0.0)
    if vac_exp > 0:
        exp_ratio = min(res_exp / vac_exp, 1.0)

    salary_match = 1.0
    res_salary = safe_float(resume.get("salary"), 0.0)
    vac_salary = safe_float(vacancy.get("salary"), 0.0)
    if res_salary and vac_salary:
        salary_match = 1.0 if res_salary <= vac_salary else max(0.0, vac_salary / res_salary)

    job_sim = cosine_sim(embed_text(vacancy.get("post","")), embed_text(resume.get("post","")))
    edu_sim = cosine_sim(embed_text(vacancy.get("education","")), embed_text(resume.get("education","")))

    vac_langs = set([l.lower() for l in vacancy.get("languages", [])])
    res_langs = set([l.lower() for l in resume.get("languages", [])])
    langs_match = len(vac_langs & res_langs) / len(vac_langs) if vac_langs else 1.0

    employment_match = 1.0 if vacancy.get("busyness") == resume.get("busyness") else 0.0

    score = (
        city_match * weights["city"] +
        exp_ratio * weights["experience"] +
        job_sim * weights["post"] +
        edu_sim * weights["education"] +
        langs_match * weights["languages"] +
        salary_match * weights["salary"] +
        employment_match * weights["busyness"]
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

    # city
    if vacancy.get("city") and resume.get("city") and vacancy["city"].lower() != resume["city"].lower():
        gaps.append({"type": "relocation", "reason": f"вакансия в {vacancy['city']}, кандидат в {resume['city']}"})

    # experience
    if d["exp_ratio"] < 0.7:
        gaps.append({"type": "training", "reason": "experience меньше требуемого"})

    # education
    if d["edu_sim"] < 0.6:
        gaps.append({"type": "education", "reason": "education не полностью совпадает"})

    # salary
    res_salary = safe_float(resume.get("salary"), 0)
    vac_salary = safe_float(vacancy.get("salary"), 0)
    if res_salary > 0 and vac_salary > 0 and res_salary > vac_salary * 1.15:
        gaps.append({"type": "salary", "reason": f"ожидания {res_salary} > оффер {vac_salary}"})

    # languages
    req_langs = set([l.lower() for l in vacancy.get("languages", [])])
    res_langs = set([l.lower() for l in resume.get("languages", [])])
    missing = list(req_langs - res_langs)
    if missing:
        gaps.append({"type": "language", "reason": f"отсутствуют требуемые languages: {', '.join(missing)}", "missing_langs": ", ".join(missing)})

    # busyness
    if vacancy.get("busyness") and resume.get("busyness") and vacancy["busyness"] != resume["busyness"]:
        gaps.append({"type": "employment", "reason": "тип занятости не совпадает"})

    # мотивация (по желанию)
    if not resume.get("мотивация") or len(resume.get("мотивация","")) < 30:
        gaps.append({"type": "motivation", "reason": "мало информации о мотивации"})

    return gaps
