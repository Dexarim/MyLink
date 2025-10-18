from typing import Dict, List, Any
from embeddings import cosine_sim, safe_float, embed_text


# === Расчёт базовой релевантности ===
def base_similarity_score(vacancy: Dict, resume: Dict) -> Dict:
    weights = {
        "город": 0.1,
        "опыт": 0.25,
        "должность": 0.25,
        "образование": 0.15,
        "языки": 0.1,
        "зарплата": 0.1,
        "занятость": 0.05
    }

    city_match = 1.0 if vacancy["город"].lower() == resume["город"].lower() else 0.0
    exp_ratio = min(safe_float(resume["опыт"]) / safe_float(vacancy["опыт"], 1.0), 1.0)
    salary_match = 1.0 if safe_float(resume["зарплата"]) <= safe_float(vacancy["зарплата"]) else 0.8

    job_sim = cosine_sim(embed_text(vacancy["должность"]), embed_text(resume["должность"]))
    edu_sim = cosine_sim(embed_text(vacancy["образование"]), embed_text(resume["образование"]))

    langs_v = set(vacancy["языки"])
    langs_r = set(resume["языки"])
    langs_match = len(langs_v & langs_r) / len(langs_v) if langs_v else 1.0

    employment_match = 1.0 if vacancy["занятость"] == resume["занятость"] else 0.0

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
            "exp_ratio": exp_ratio,
            "job_sim": job_sim,
            "edu_sim": edu_sim,
            "langs_match": langs_match,
            "salary_match": salary_match,
            "employment_match": employment_match
        }
    }



# === Анализ несоответствий ===
def detect_gaps(vacancy: Dict, resume: Dict, scoring: Dict) -> List[Dict]:
    gaps = []
    d = scoring["details"]

    if vacancy["город"].lower() != resume["город"].lower():
        gaps.append({"type": "relocation", "reason": f"вакансия в {vacancy['город']}, а кандидат в {resume['город']}"})

    if d["exp_ratio"] < 0.7:
        gaps.append({"type": "training", "reason": f"опыт меньше требуемого"})

    if d["edu_sim"] < 0.6:
        gaps.append({"type": "education", "reason": "образование не полностью совпадает"})

    return gaps
