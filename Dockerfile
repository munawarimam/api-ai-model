FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install libsndfile1 -y

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 80

CMD ["uvicorn", "API.main:app", "--host", "0.0.0.0", "--port", "80"]
