# api_server.py
import os
from fastapi import FastAPI, Depends, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session, joinedload
from typing import List, Optional

from database import Base, engine, SessionLocal
from models import Resume, Vacancy, Application
import schemas

# ---------- FastAPI ----------
app = FastAPI(
    title="Resume Intake API",
    description="Отдельный API для приёма и хранения резюме/вакансий)",
    version="1.0.0",
)

# ---------- CORS ----------
origins_env = os.getenv("CORS_ORIGINS", "*")
allow_origins = [o.strip() for o in origins_env.split(",")] if origins_env else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- DB ----------
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ================== VACANCIES ==================
@app.post("/vacancies/", response_model=schemas.VacancyOut)
def create_vacancy(v: schemas.VacancyCreate, db: Session = Depends(get_db)):
    obj = Vacancy(**v.dict())
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


@app.get("/vacancies/", response_model=List[schemas.VacancyOut])
def list_vacancies(db: Session = Depends(get_db),
                   city: str = Query(default=None),
                   q: str = Query(default=None)):
    query = db.query(Vacancy)
    if city:
        query = query.filter(Vacancy.city.ilike(f"%{city}%"))
    if q:
        query = query.filter(Vacancy.post.ilike(f"%{q}%"))
    return query.order_by(Vacancy.id.desc()).all()


@app.get("/vacancies/{vacancy_id}", response_model=schemas.VacancyOut)
def get_vacancy(vacancy_id: int = Path(..., ge=1), db: Session = Depends(get_db)):
    obj = db.get(Vacancy, vacancy_id)
    if not obj:
        raise HTTPException(404, "Вакансия не найдена")
    return obj


@app.delete("/vacancies/{vacancy_id}")
def delete_vacancy(vacancy_id: int, db: Session = Depends(get_db)):
    obj = db.get(Vacancy, vacancy_id)
    if not obj:
        raise HTTPException(404, "Вакансия не найдена")
    db.delete(obj)
    db.commit()
    return {"ok": True}

# ================== RESUMES ==================
@app.post("/resumes/", response_model=schemas.ResumeOut)
def create_resume(r: schemas.ResumeCreate, db: Session = Depends(get_db)):
    obj = Resume(
        name=r.name, email=r.email, phone=r.phone,
        city=r.city, experience=r.experience, post=r.post,
        education=r.education, languages=r.languages, salary=r.salary,
        busyness=r.busyness, skills=r.skills, motivation=r.motivation,
    )
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


@app.get("/resumes/", response_model=List[schemas.ResumeOut])
def list_resumes(db: Session = Depends(get_db),
                 city: str = Query(default=None),
                 q: str = Query(default=None)):
    query = db.query(Resume)
    if city:
        query = query.filter(Resume.city.ilike(f"%{city}%"))
    if q:
        query = query.filter(Resume.post.ilike(f"%{q}%"))
    return query.order_by(Resume.id.desc()).all()


@app.get("/resumes/{resume_id}", response_model=schemas.ResumeOut)
def get_resume(resume_id: int = Path(..., ge=1), db: Session = Depends(get_db)):
    obj = db.get(Resume, resume_id)
    if not obj:
        raise HTTPException(404, "Резюме не найдено")
    return obj


@app.delete("/resumes/{resume_id}")
def delete_resume(resume_id: int, db: Session = Depends(get_db)):
    obj = db.get(Resume, resume_id)
    if not obj:
        raise HTTPException(404, "Резюме не найдено")
    db.delete(obj)
    db.commit()
    return {"ok": True}

# ================== APPLICATIONS ==================
@app.post("/applications/", response_model=schemas.ApplicationOut)
def create_application(payload: schemas.ApplicationCreate, db: Session = Depends(get_db)):
    resume = db.get(Resume, payload.resume_id)
    if not resume:
        raise HTTPException(404, "Резюме не найдено")

    vacancy = db.get(Vacancy, payload.vacancy_id)
    if not vacancy:
        raise HTTPException(404, "Вакансия не найдена")

    app_row = Application(resume_id=resume.id, vacancy_id=vacancy.id, in_outbox=True)
    db.add(app_row)
    db.commit()
    db.refresh(app_row)

    app_row = (
        db.query(Application)
        .options(joinedload(Application.resume), joinedload(Application.vacancy_obj))
        .filter(Application.id == app_row.id)
        .first()
    )
    return schemas.ApplicationOut(
        id=app_row.id,
        resume=app_row.resume,
        vacancy=app_row.vacancy_obj,
        in_outbox=app_row.in_outbox,
    )


@app.get("/applications/{app_id}", response_model=schemas.ApplicationOut)
def get_application(app_id: int, db: Session = Depends(get_db)):
    app_row = (
        db.query(Application)
        .options(joinedload(Application.resume), joinedload(Application.vacancy_obj))
        .filter(Application.id == app_id).first()
    )
    if not app_row:
        raise HTTPException(404, "Заявка не найдена")
    return schemas.ApplicationOut(
        id=app_row.id,
        resume=app_row.resume,
        vacancy=app_row.vacancy_obj,
        in_outbox=app_row.in_outbox,
    )


@app.delete("/applications/{app_id}")
def delete_application(app_id: int, db: Session = Depends(get_db)):
    app_row = db.get(Application, app_id)
    if not app_row:
        raise HTTPException(404, "Заявка не найдена")
    db.delete(app_row)
    db.commit()
    return {"ok": True}

# ================== OUTBOX ==================
@app.get("/outbox/next", response_model=Optional[schemas.OutboxItem])
def outbox_next(db: Session = Depends(get_db)):
    app_row = (
        db.query(Application)
        .options(joinedload(Application.resume), joinedload(Application.vacancy_obj))
        .filter(Application.in_outbox == True)
        .order_by(Application.id.asc())
        .first()
    )
    if not app_row:
        return None

    app_row.in_outbox = False
    db.commit()
    db.refresh(app_row)

    return schemas.OutboxItem(
        application_id=app_row.id,
        resume=app_row.resume,
        vacancy=app_row.vacancy_obj,
    )


@app.post("/outbox/{application_id}/reset")
def outbox_reset(application_id: int, db: Session = Depends(get_db)):
    app_row = db.get(Application, application_id)
    if not app_row:
        raise HTTPException(404, "Заявка не найдена")
    app_row.in_outbox = True
    db.commit()
    return {"ok": True}

# ---------- root ----------
@app.get("/")
def root():
    return {"ok": True, "service": "Resume Intake API", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=int(os.getenv("PORT", 8002)), reload=True)
