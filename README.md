AI Model Speech to Text

### Quick Start in local

```
Please insert the secret variable in .env file, contains db connection and your secret key for encode the credentials

$ docker compose up -d

or you can run it without docker, make sure you have put the connection and secret key in your local

$ pip install -r requirements.txt
$ uvicorn API.main:app --reload
```

### Unit Test
```
$ pytest
```

### Result
Swagger UI, you can access it on http://localhost/api/v1/docs#/