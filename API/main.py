from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, UploadFile, File, Form
from typing import Annotated
from sqlalchemy.orm import Session
from auth.db import engine, session
import auth.authentication as auth
from starlette import status
from auth.authentication import get_current_user
from models.models import Base, MLModel, CreateListModelRequest, Job, STTResult, SAResult
from datetime import datetime, timezone
import yaml
from importlib import import_module

tags_metadata = [
    {
        "name": "users",
        "description": "Operations with users. Get user id",
    },
    {
        "name": "models",
        "description": "Create new model into lists and listing model that available",
    },
    {
        "name": "auth",
        "description": "Create new user and get token for send a request"
    },
]

app = FastAPI(root_path="/api/v1", openapi_tags=tags_metadata)
app.include_router(auth.router)

Base.metadata.create_all(bind=engine)

def get_db():
    db = session()
    try:
        yield db
    finally:
        db.close()

def load_model_config(config_path='config/config_model.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_model_function(config, db: Session, model_id: int, **params):
    model = db.query(MLModel).filter(MLModel.id == model_id).first()
    if not model:
        raise ValueError(f"No model found for model_id {model_id}")
    
    for model_config in config['models']:
        if model_config['name'] == model.ml_model_name:
            module_name = model_config['module']
            class_name, function_name = model_config['function'].rsplit('.', 1)
            module = import_module(f'scripts.{module_name}')
            model_class = getattr(module, class_name)
            model_params = {key: value for key, value in params.items() if key in model_config['params']}  
            model_instance = model_class(**model_params)
            function = getattr(model_instance, function_name)
            return function

def get_desc_result_model_id(config, db: Session, model_id: int):
    model = db.query(MLModel).filter(MLModel.id == model_id).first()
    if not model:
        raise ValueError(f"No model found for model_id {model_id}")
    
    for model_config in config['models']:
        if model_config['name'] == model.ml_model_name:
            table_model = globals()[model_config['table_model']]
            output_columns = model_config['output_columns']
            return table_model, output_columns
        else:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="models has not registered to the system yet")

db_dependency = Annotated[Session, Depends(get_db)]
user_dependency = Annotated[dict, Depends(get_current_user)]
model_config = load_model_config()

@app.get("/user", status_code=status.HTTP_200_OK, tags=["users"])
async def check_user(user: user_dependency):
    if user is None:
        raise HTTPException(status_code=401, detail='Authentication Failed')
    return {"User": user}

def process_audio(db: Session, job_id: str, model_id: int, correlation_id: str, **params):
    try:
        model_function = get_model_function(model_config, db, model_id, **params)
        if not model_function:
            raise ValueError(f"No model function found for model_id {model_id}")
        
        model_function(job_id=job_id, model_id=model_id, correlation_id=correlation_id)

        job = db.query(Job).filter(Job.id == job_id).first()
        job.complete = True
        job.message = "successful"
        job.updated_at = datetime.now(tz=timezone.utc)
        db.commit()
    except Exception as e:
        job = db.query(Job).filter(Job.id == job_id).first()
        job.complete = True
        job.message = f"failed: {str(e)}"
        job.updated_at = datetime.now(tz=timezone.utc)
        db.commit()

@app.post("/models/regis_model", status_code=status.HTTP_201_CREATED, tags=["models"])
async def create_model_name(db: db_dependency, user: user_dependency,
                      create_model_request: CreateListModelRequest):
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication Failed')
    create_ml_model = MLModel(
        ml_model_name = create_model_request.ml_model_name)
    db.add(create_ml_model)
    db.commit()

@app.get("/models", status_code=status.HTTP_200_OK, tags=["models"])
async def list_models(db: db_dependency, user: user_dependency):
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication Failed')
    
    models = db.query(MLModel).all()
    return [{"id": model.id, "ml_model_name": model.ml_model_name} for model in models]

@app.post("/models/{model_id}/inference", status_code=status.HTTP_201_CREATED, tags=["models"])
async def create_inference(model_id: int, background_tasks: BackgroundTasks,
                           db: db_dependency, user: user_dependency,
                           data: UploadFile = File(...), explaining: str = Form(...), correlation_id: str = Form(...)):
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication Failed')
    
    if explaining.lower() != "true":
        raise HTTPException(status_code=400, detail="Invalid value for 'explaining'")
    
    if not (data.filename.endswith(".mp3") or data.filename.endswith(".wav")):
        raise HTTPException(status_code=400, detail="Invalid file format. Only mp3 and wav files are supported.")

    file_contents = data.file.read()

    job = Job(
        model_id=model_id,
        correlation_id=correlation_id,
        transaction="reply",
        complete=False,
        message="on progress",
        file_name=data.filename 
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    
    background_tasks.add_task(process_audio, db, job.id, model_id, correlation_id, audio_contents=file_contents)
    
    return {"message": "created", "job_id": job.id}

@app.get("/models/{model_id}/responses", status_code=status.HTTP_200_OK, tags=["models"])
async def check_inference_status(model_id: int, type: str, correlation_id: str, db: db_dependency, user: user_dependency):
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication Failed')

    if type != "inference":
        raise HTTPException(status_code=400, detail="Invalid value for 'type'")
    
    jobs = db.query(Job).filter(Job.model_id == model_id, Job.correlation_id == correlation_id).all()
    responses = []
    for job in jobs:
        response = {
            "id": job.id,
            "model_id": job.model_id,
            "transaction": job.transaction,
            "updated_at": job.updated_at.isoformat(),
            "correlation_id": job.correlation_id,
            "progress": {
                "complete": job.complete,
                "message": job.message,
            },
            "file_name": job.file_name 
        }
        responses.append(response)
    
    return responses

@app.get("/models/{model_id}/results", status_code=status.HTTP_200_OK, tags=["models"])
async def get_results(model_id: int, correlation_id: str, db: db_dependency, user: user_dependency):
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication Failed')
    
    table_model, output_columns = get_desc_result_model_id(model_config, db, model_id)
    results = db.query(table_model).filter(table_model.model_id == model_id, table_model.correlation_id == correlation_id).all()
    
    if not results:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='No records found for the given correlation_id')
    
    responses = []
    for result in results:
        result_dict = {col: getattr(result, col) for col in output_columns}
        responses.append(result_dict)
    
    return responses