## Speech-to-Text and Stress Analysis API
This repository contains an API that leverages advanced AI models for speech-to-text conversion and stress analysis. The API allows users to upload audio files in MP3 and WAV formats. It processes the audio to provide two key features:

1. Speech-to-Text Conversion: Transcribes the spoken content in the audio into text.
2. Stress Analysis: Analyzes the audio for stress indicators and provides insights based on vocal patterns.

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

#### 1. Swagger UI, you can access it on http://localhost/api/v1/docs#/

![alt text](https://github.com/munawarimam/api-ai-model/blob/main/results/Screenshot%202024-07-31%20at%2013.04.03.png?raw=true)

#### 2. Create User

![alt text](https://github.com/munawarimam/api-ai-model/blob/main/results/Screenshot%202024-07-31%20at%2015.40.10.png?raw=true)

#### 3. Get Token

![alt text](https://github.com/munawarimam/api-ai-model/blob/main/results/Screenshot%202024-07-31%20at%2013.11.54.png?raw=true)

#### 4. Token Expired

![alt text](https://github.com/munawarimam/api-ai-model/blob/main/results/Screenshot%202024-07-31%20at%2013.16.17.png?raw=true)

#### 5. Models

![alt text](https://github.com/munawarimam/api-ai-model/blob/main/results/Screenshot%202024-07-31%20at%2013.23.59.png?raw=true)

#### 6. Transcript Audio

![alt text](https://github.com/munawarimam/api-ai-model/blob/main/results/Screenshot%202024-07-31%20at%2014.49.23.png?raw=true)

#### 7. Check Status Transcript

![alt text](https://github.com/munawarimam/api-ai-model/blob/main/results/Screenshot%202024-07-31%20at%2014.49.36.png?raw=true)

#### 8. Get Result

![alt text](https://github.com/munawarimam/api-ai-model/blob/main/results/Screenshot%202024-07-31%20at%2014.49.51.png?raw=true)

