import time
import whisper
import soundfile as sf
import pandas as pd
import librosa
from io import BytesIO

from datetime import datetime, timezone

from auth.db import engine

class TranscriptionGenerator:
    """
    A class that generates transcriptions from audio files using the Whisper ASR model.

    Attributes:
        model: The Whisper ASR model used for transcription generation.

    Methods:
        __init__(self): Initializes the TranscriptionGenerator object.
        generate_transcription(self, audio_path): Generates a transcription from the given audio file.
        read_audio_and_generate_transcription(self, audio_path): Reads an audio file and generates a transcription.
    """

    def __init__(self, audio_contents):
        self.model = whisper.load_model('medium')
        file_buffer = BytesIO(audio_contents)
        self.data_buffer, self.samplerate = librosa.load(file_buffer)  
    
    def generate_transcription(self):
        """
        Generates a transcription from the given audio file.

        Args:
            audio_path (str): The path to the audio file.

        Returns:
            str: The generated transcription.
        """
        result = self.model.transcribe(audio=self.data_buffer, language='id', verbose=True)
        transcription_text = result.get('text', '')
        trimmed_text = '{' + ' '.join(transcription_text.strip().split()).replace("'", "") + '}'
        return trimmed_text

    def get_audio_duration(self):
        """
        Get the duration of the audio file.

        Args:
            audio_path (str): The path to the audio file.

        Returns:
            float: The duration of the audio file in seconds.
        """
        duration = round((len(self.data_buffer) / self.samplerate)/60, 2)
        return duration

    def transcribe(self, job_id, model_id, correlation_id):
        """
        Reads an audio file and generates a transcription.

        Args:
            audio_path (str): The path to the audio file.

        Returns:
            str: The generated transcription.

        Raises:
            ValueError: If the audio file format is invalid. Only mp3 and wav files are supported.
        """
        startimestamp   = int(time.time())
        start_time      = str(datetime.now(tz=timezone.utc))[:10] + 'T' + str(datetime.now(tz=timezone.utc))[11:19]
        transcription   = self.generate_transcription()
        audio_duration  = self.get_audio_duration()
        finish_time     = str(datetime.now(tz=timezone.utc))[:10] + 'T' + str(datetime.now(tz=timezone.utc))[11:19]
        duration        = round((int(time.time()) - startimestamp) / 60, 2)
        df              = pd.DataFrame([[job_id, model_id, correlation_id, transcription, audio_duration, start_time, finish_time, duration, datetime.now(tz=timezone.utc)]], 
                        columns=['job_id', 'model_id', 'correlation_id', 'transcription', 'audio_duration', 'start_time', 'finish_time', 'stt_duration', 'inserted_at'])
        df.to_sql("stt_result", engine, if_exists='append', index=False)