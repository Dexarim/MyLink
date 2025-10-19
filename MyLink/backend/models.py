# models.py
from sqlalchemy import Column, Integer, String, Float, JSON, Text, ForeignKey, DateTime, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base

class Resume(Base):
    __tablename__ = "resumes"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=True)
    email = Column(String, nullable=True, index=True)
    phone = Column(String, nullable=True)

    city = Column(String, nullable=False)
    experience = Column(Float, nullable=False)
    post = Column(String, nullable=False)
    education = Column(String, nullable=False)
    languages = Column(JSON, nullable=False)
    salary = Column(Float, nullable=False)
    busyness = Column(String, nullable=False)
    skills = Column(JSON, nullable=False)
    motivation = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    applications = relationship("Application", back_populates="resume", cascade="all, delete-orphan")


class Vacancy(Base):
    __tablename__ = "vacancies"

    id = Column(Integer, primary_key=True, index=True)
    city = Column(String, nullable=False)
    experience = Column(Float, nullable=False)
    post = Column(String, nullable=False)
    education = Column(String, nullable=False)
    languages = Column(JSON, nullable=False)
    salary = Column(Float, nullable=False)
    busyness = Column(String, nullable=False)
    skills = Column(JSON, nullable=False)
    description = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    applications = relationship("Application", back_populates="vacancy_obj", cascade="all, delete-orphan")


class Application(Base):
    __tablename__ = "applications"

    id = Column(Integer, primary_key=True, index=True)
    resume_id = Column(Integer, ForeignKey("resumes.id", ondelete="CASCADE"), nullable=False)
    vacancy_id = Column(Integer, ForeignKey("vacancies.id", ondelete="CASCADE"), nullable=False)

    in_outbox = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    resume = relationship("Resume", back_populates="applications")
    vacancy_obj = relationship("Vacancy", back_populates="applications")
