"""
ocr_agent.py
Image OCR Agent - extracts text from an uploaded image (Arabic + English)
using Tesseract via pytesseract.

Tesseract itself is NOT a Python package - it's a separate program that
has to be installed on the machine (see README). If it isn't on your
PATH, set TESSERACT_CMD in .env to the full path of tesseract.exe.
"""
import io
import os

from dotenv import load_dotenv
from PIL import Image
import pytesseract

load_dotenv()

TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# "ara+eng" reads both Arabic and English in the same image.
OCR_LANGUAGES = os.getenv("OCR_LANGUAGES", "ara+eng")


class OCRAgent:
    def extract_text(self, image_bytes: bytes) -> str:
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image, lang=OCR_LANGUAGES)
        return text.strip()
