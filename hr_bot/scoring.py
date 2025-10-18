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



def detect_gaps(vacancy: Dict, resume: Dict, scoring: Dict) -> List[Dict]:
    """
    Анализирует расхождения между вакансией и резюме.
    Использует как численные метрики (из scoring), так и прямые проверки данных.
    """
    gaps = []
    d = scoring.get("details", {})

    # === 1. Город ===
    if vacancy.get("город") and resume.get("город"):
        if vacancy["город"].lower() != resume["город"].lower():
            gaps.append({
                "type": "relocation",
                "reason": f"вакансия в {vacancy['город']}, а кандидат проживает в {resume['город']}"
            })

    # === 2. Опыт работы ===
    # Учитываем как прямое значение, так и числовой показатель exp_ratio
    if d.get("exp_ratio", 1.0) < 0.7 or resume.get("опыт_лет", 0) < vacancy.get("требуемый_опыт", 0):
        gaps.append({
            "type": "training",
            "reason": "опыт кандидата меньше требуемого по вакансии"
        })

    # === 3. Образование ===
    if d.get("edu_sim", 1.0) < 0.6 or (
        vacancy.get("образование") and resume.get("образование") and
        vacancy["образование"].lower() not in resume["образование"].lower()
    ):
        gaps.append({
            "type": "education",
            "reason": "уровень или профиль образования отличается от требований"
        })

    # === 4. Зарплата ===
    if resume.get("зарплата") and vacancy.get("зарплата"):
        if resume["зарплата"] > vacancy["зарплата"]:
            gaps.append({
                "type": "salary",
                "reason": f"ожидания по зарплате ({resume['зарплата']}₸) выше предложенной ({vacancy['зарплата']}₸)"
            })

    # === 5. Языки ===
    if "языки" in vacancy and "языки" in resume:
        missing_langs = [lang for lang in vacancy["языки"] if lang not in resume["языки"]]
        if missing_langs:
            gaps.append({
                "type": "language",
                "reason": f"отсутствуют требуемые языки: {', '.join(missing_langs)}",
                "missing_langs": ", ".join(missing_langs)
            })

    # === 6. Тип занятости ===
    if vacancy.get("занятость") and resume.get("занятость"):
        if vacancy["занятость"].lower() != resume["занятость"].lower():
            gaps.append({
                "type": "employment",
                "reason": f"вакансия требует {vacancy['занятость']}, а кандидат предпочитает {resume['занятость']}"
            })

    # === 7. Мотивация ===
    if not resume.get("мотивация") or len(resume["мотивация"].strip()) < 30:
        gaps.append({
            "type": "motivation",
            "reason": "в резюме отсутствует информация о мотивации или профессиональных целях"
        })

    return gaps

