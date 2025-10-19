# schemas.py
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional

# ---------- Vacancy ----------
class VacancyBase(BaseModel):
    city: str
    experience: float
    post: str
    education: str
    languages: List[str]
    salary: float
    busyness: str
    skills: List[str]
    description: Optional[str] = None


class VacancyCreate(VacancyBase):
    pass


class VacancyOut(VacancyBase):
    id: int

    class Config:
        from_attributes = True


# ---------- Resume ----------
class ResumeBase(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None

    city: str
    experience: float
    post: str
    education: str
    languages: List[str]
    salary: float
    busyness: str
    skills: List[str]
    motivation: Optional[str] = Field(default=None, alias="мотивация")

    class Config:
        populate_by_name = True


class ResumeCreate(ResumeBase):
    pass


class ResumeOut(ResumeBase):
    id: int

    class Config:
        from_attributes = True


# ---------- Application ----------
class ApplicationCreate(BaseModel):
    resume_id: int
    vacancy_id: int


class ApplicationOut(BaseModel):
    id: int
    resume: ResumeOut
    vacancy: VacancyOut
    in_outbox: bool

    class Config:
        from_attributes = True


# ---------- Outbox ----------
class OutboxItem(BaseModel):
    application_id: int
    resume: ResumeOut
    vacancy: VacancyOut
