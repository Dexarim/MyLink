import json
import requests
import numpy as np
from typing import Dict, List, Any
from functools import lru_cache
from sentence_transformers import SentenceTransformer, util

# === Инициализация модели эмбеддингов ===
EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
EMB_DIM = EMB_MODEL.get_sentence_embedding_dimension()


# === Утилиты ===
def safe_float(value, default=0.0):
    try:
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_vec(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / (norm + 1e-12)


def embed_text(text: str) -> np.ndarray:
    if not text:
        return np.zeros(EMB_DIM, dtype=float)
    emb = EMB_MODEL.encode(text, convert_to_numpy=True)
    return normalize_vec(emb)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# === Клиент для Mistral/Ollama ===
class MistralClient:
    def __init__(self, model="mistral", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    @lru_cache(maxsize=128)
    def generate(self, prompt: str) -> str:
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt},
                stream=True,
                timeout=30
            )
            text = ""
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line.decode("utf-8"))
                text += data.get("response", "")
            return text.strip()
        except Exception as e:
            return f"[Ошибка Mistral: {e}]"


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
    Интерпретация сложных ответов по смыслу.
    Сравнивает ответ кандидата с примерами положительных и отрицательных фраз.
    """
    positive_samples = [
        "да", "конечно", "готов", "без проблем", "можно", "согласен", "буду рад", "не против", "да, конечно",
        "готов пройти обучение", "готов переехать", "готов учиться"
    ]
    negative_samples = [
        "нет", "не могу", "не готов", "не хочу", "не согласен", "нет возможности", "это невозможно", "я не уверен"
    ]

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
        return None  # Неопределённый ответ


# === Пересчёт итогового результата ===
def integrate_followup(vacancy, resume, scoring, followups):
    details = scoring["details"].copy()
    for f in followups:
        accepted = interpret_answer(f["answer"])
        if accepted and f["gap"]["type"] == "relocation":
            details["city_match"] = 1.0
        elif accepted and f["gap"]["type"] == "training":
            details["exp_ratio"] = min(1.0, details["exp_ratio"] + 0.2)
        elif accepted and f["gap"]["type"] == "education":
            details["edu_sim"] = min(1.0, details["edu_sim"] + 0.15)

    new_score = (
        details["city_match"] * 0.1 +
        details["exp_ratio"] * 0.25 +
        details["job_sim"] * 0.25 +
        details["edu_sim"] * 0.15 +
        details["langs_match"] * 0.1 +
        details["salary_match"] * 0.1 +
        details["employment_match"] * 0.05
    ) * 100

    return round(new_score, 2)


# === Основной сценарий ===
if __name__ == "__main__":
    vacancy = {
        "город": "Астане",
        "опыт": 5,
        "должность": "junior python developer",
        "образование": "высшее специальное",
        "языки": ["английский", "русский"],
        "зарплата": 800000,
        "занятость": "частичное"
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
