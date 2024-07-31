from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Float, PrimaryKeyConstraint, JSON
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import Sequence
from pydantic import BaseModel
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime

Base = declarative_base()

class Users(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True)
    hashed_password = Column(String)

class CreateUserRequest(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class MLModel(Base):
    __tablename__ = 'ml_models'
    id = Column(Integer, Sequence('ml_model_seq', start=1001, increment=1), primary_key=True)
    ml_model_name = Column(String(100), unique=True)

class CreateListModelRequest(BaseModel):
    ml_model_name: str

class Job(Base):
    __tablename__ = 'ml_models_inference'

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey('ml_models.id'))
    file_name = Column(String)
    correlation_id = Column(String, index=True)
    transaction = Column(String)
    complete = Column(Boolean, default=False)
    message = Column(String, nullable=True)
    updated_at = Column(DateTime, default=datetime.now)

    model = relationship('MLModel')

class STTResult(Base):
    __tablename__ = 'stt_result'
    __table_args__ = (
        PrimaryKeyConstraint('job_id', 'model_id'),
    )

    job_id = Column(Integer, ForeignKey('ml_models_inference.id'))
    model_id = Column(Integer, ForeignKey('ml_models.id'))
    correlation_id = Column(String, index=True)
    transcription = Column(JSON)
    audio_duration = Column(Float)
    start_time = Column(DateTime)
    finish_time = Column(DateTime)
    stt_duration = Column(Integer)
    inserted_at = Column(DateTime)

    model = relationship('MLModel')
    job = relationship('Job')


class SAResult(Base):
    __tablename__ = 'sa_result'
    __table_args__ = (
        PrimaryKeyConstraint('job_id', 'model_id'),
    )

    job_id = Column(Integer, ForeignKey('ml_models_inference.id'))
    model_id = Column(Integer, ForeignKey('ml_models.id'))
    correlation_id = Column(String, index=True)
    emotion_result = Column(String)
    confidence_value = Column(Float)
    audio_duration = Column(Float)
    start_time = Column(DateTime)
    finish_time = Column(DateTime)
    sa_duration = Column(Integer)
    inserted_at = Column(DateTime)

    model = relationship('MLModel')
    job = relationship('Job')