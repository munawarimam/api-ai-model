import librosa
import torch
import time
import pandas as pd
import soundfile as sf
import torch.nn as nn
import torch.nn.functional as F
import librosa
from io import BytesIO

from datetime import datetime, timezone
from transformers import AutoConfig, Wav2Vec2FeatureExtractor, HubertPreTrainedModel, HubertModel
from auth.db import engine


model_name  = "xmj2002/hubert-base-ch-speech-emotion-recognition"
duration    = 15
sample_rate = 16000
model_id    = 150

config = AutoConfig.from_pretrained(
    pretrained_model_name_or_path=model_name,
)



class HubertForSpeechClassification(HubertPreTrainedModel):
    """
    Hubert model for speech classification.

    Args:
        config (HubertConfig): The model configuration class instance.

    Attributes:
        hubert (HubertModel): The Hubert model.
        classifier (HubertClassificationHead): The classification head.
    """

    def __init__(self, config):
        super().__init__(config)
        self.hubert = HubertModel(config)
        self.classifier = HubertClassificationHead(config)
        self.init_weights()

    def forward(self, x):
        """
        Forward pass of the HubertForSpeechClassification model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        outputs = self.hubert(x)
        hidden_states = outputs[0]
        x = torch.mean(hidden_states, dim=1)
        x = self.classifier(x)
        return x


class HubertClassificationHead(nn.Module):
    def __init__(self, config):
        """
        Initializes the HubertClassificationHead module.

        Args:
            config (object): Configuration object containing the model's hyperparameters.
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_class)

    def forward(self, x):
        """
        Performs forward pass of the HubertClassificationHead module.

        Args:
            x (tensor): Input tensor.

        Returns:
            tensor: Output tensor.
        """
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class StressAnalysisGenerator:
    """
    Class for generating stress analysis results from audio files.
    """

    def __init__(self, audio_contents):
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = HubertForSpeechClassification.from_pretrained(
            model_name,
            config=config,
        )
        file_buffer = BytesIO(audio_contents)
        self.data_buffer, self.samplerate = librosa.load(file_buffer, sr=sample_rate)
        

    def id2class(self, id):
        """
        Convert class ID to corresponding emotion class.

        Args:
            id (int): Class ID.

        Returns:
            str: Emotion class.
        """
        if id == 0:
            return "angry"
        elif id == 1:
            return "fear"
        elif id == 2:
            return "happy"
        elif id == 3:
            return "neutral"
        elif id == 4:
            return "sadness"
        else:
            return "excited"


    def predict(self):
        """
        Predict the emotion class of an audio file.

        Args:
            audio_path (str): Path to the audio file.
            processor: Feature extractor for audio processing.
            model: Speech classification model.

        Returns:
            str: Emotion class prediction.
        """
        speech      = self.processor(self.data_buffer, padding="max_length", truncation=True, max_length=duration * self.samplerate, return_tensors="pt", sampling_rate=self.samplerate).input_values

        with torch.no_grad():
            logit = self.model(speech)

        score   = F.softmax(logit, dim=1).detach().cpu().numpy()[0]
        id      = torch.argmax(logit).cpu().numpy()

        return self.id2class(id), score[id]

    def get_audio_duration(self):
        """
        Get the duration of an audio file.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            float: Duration of the audio file in minutes.
        """
        duration = round((len(self.data_buffer) / self.samplerate)/60, 2)
        return duration


    def transcribe(self, job_id, correlation_id, model_id):
        """
        Transcribe an audio file and store the results in a database.

        Args:
            audio_path (str): Path to the audio file.
            correlation_id (str): Correlation ID for tracking purposes.

        Raises:
            ValueError: If the audio file format is not supported.

        """
        startimestamp   = int(time.time())
        start_time      = str(datetime.now(tz=timezone.utc))[:10] + 'T' + str(datetime.now(tz=timezone.utc))[11:19]
        emotion_result, confidence_value  = self.predict()
        audio_duration  = self.get_audio_duration()
        finish_time     = str(datetime.now(tz=timezone.utc))[:10] + 'T' + str(datetime.now(tz=timezone.utc))[11:19]
        duration        = round((int(time.time()) - startimestamp) / 60, 2)
        df              = pd.DataFrame([[job_id, model_id, correlation_id, emotion_result, round(confidence_value, 2), audio_duration, start_time, finish_time, duration, datetime.now(tz=timezone.utc)]], 
                        columns=['job_id', 'model_id', 'correlation_id', 'emotion_result', 'confidence_value', 'audio_duration', 'start_time', 'finish_time', 'sa_duration', 'inserted_at'])
        df.to_sql("sa_result", engine, if_exists='append', index=False)