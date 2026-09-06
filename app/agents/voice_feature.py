"""
voice_agent.py
Voice Query Agent - turns a recorded voice clip into text using OpenAI's
Whisper transcription API. Uses the same OpenAI key as the Answer Agent
(API_KEY in .env) - no extra key needed.
"""
import io
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("API_KEY")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-1")


class VoiceAgent:
    def __init__(self):
        self.client = OpenAI(api_key=API_KEY)

    def transcribe(self, audio_bytes: bytes, filename: str = "audio.webm") -> str:
        """
        audio_bytes: raw bytes of a recorded clip (webm/mp3/wav/m4a...)
        filename: only used so the SDK can tell what format the bytes are in
        returns: the transcribed text (Whisper auto-detects the language,
                 so this works for Arabic and English without extra setup)
        """
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = filename

        transcript = self.client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=audio_file,
        )
        return transcript.text.strip()
