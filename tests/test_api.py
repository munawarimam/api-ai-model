import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from API.main import app, get_db
from models.models import Base, Users
from passlib.context import CryptContext
from jose import jwt
from datetime import datetime, timedelta

SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)

bcrypt_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

SECRET_KEY = '3fc6e14711a50f4b2cdabb8e32645c9c7d4e7a5e72174dbb3e454ccccbc4dbaa'
ALGORITHM = 'HS256'

def create_access_token(username: str, user_id: int, expires_delta: timedelta):
    encode = {'username': username, 'id': user_id}
    expires = datetime.now() + expires_delta
    encode.update({'exp': expires})
    return jwt.encode(encode, SECRET_KEY, algorithm=ALGORITHM)

@pytest.fixture(scope="module", autouse=True)
def create_new_user():
    db = TestingSessionLocal()
    hashed_password = bcrypt_context.hash("testpassword")
    new_user = Users(username="testuser", hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.close()
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def access_token():
    db = TestingSessionLocal()
    user = db.query(Users).filter(Users.username == "testuser").first()
    token = create_access_token(user.username, user.id, timedelta(minutes=20))
    db.close()
    return token

def test_create_user():
    response = client.post(
        "/auth/",
        json={"username": "testuser", "password": "testpassword"}
    )
    assert response.status_code in [201, 400]

def test_login_for_access_token():
    response = client.post(
        "/auth/token",
        headers={"Authorization": "Basic dGVzdHVzZXI6dGVzdHBhc3N3b3Jk"}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"

def test_check_user(access_token):
    response = client.get(
        "/user",
        headers={"Authorization": f"Bearer {access_token}"}
    )
    assert response.status_code == 200
    assert response.json() == {"User": {"username": "testuser", "id": 1}}

def test_create_model_name(access_token):
    response = client.post(
        "/models/regis_model",
        headers={"Authorization": f"Bearer {access_token}"},
        json={"ml_model_name": "test_model"}
    )
    assert response.status_code == 201

def test_list_models(access_token):
    response = client.get(
        "/models",
        headers={"Authorization": f"Bearer {access_token}"}
    )
    assert response.status_code == 200
    assert any(model["ml_model_name"] == "test_model" for model in response.json())

def test_create_inference(access_token):
    data = {
        "explaining": "true",
        "correlation_id": "test_correlation_id"
    }
    files = {
        "data": ("../data/audio/02_30-0.wav", b"some file content", "audio/wav")
    }
    response = client.post(
        "/models/1/inference",
        headers={"Authorization": f"Bearer {access_token}"},
        data=data,
        files=files
    )
    assert response.status_code == 201
    assert "message" in response.json()
    assert "job_id" in response.json()

def test_check_inference_status(access_token):
    response = client.get(
        "/models/1/responses",
        headers={"Authorization": f"Bearer {access_token}"},
        params={"type": "inference", "correlation_id": "test_correlation_id"}
    )
    assert response.status_code == 200

def test_get_results(access_token):
    response = client.get(
        "/models/1/results",
        headers={"Authorization": f"Bearer {access_token}"},
        params={"correlation_id": "test_correlation_id"}
    )
    assert response.status_code in [200, 404]