"""
main.py
FastAPI backend for the Multi-Agent RAG Document Analyzer.

Run from the project root with:
    uvicorn app.main:app --reload

Endpoints:
- POST /api/chat        {"question": "..."}      -> {"id": int, "answer": "..."}
- POST /api/ocr         image file (upload)       -> {"text": "..."}
(voice query runs entirely in the browser via the Web Speech API - no
server endpoint needed for it)
- GET  /api/report/{id} builds+downloads a PDF for a past answer
- POST /api/reset       clears chat history
- GET  /api/health      {"ready": true|false}
- GET  /                the chat UI (static/index.html)
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .monitoring import setup_langsmith
from .orchestrator import Orchestrator
from .agents.image_to_text_feature import OCRAgent

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# Holds the single Orchestrator instance once it's built (heavy - loads
# the embedding + reranker models, so this only happens once, at startup).
state = {"orchestrator": None}

# Lightweight agent - just a thin wrapper around Tesseract, no heavy model
# loading, so it's fine to create right away instead of waiting for lifespan.
ocr_agent = OCRAgent()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_langsmith()
    print("🔄 Loading models and building the Orchestrator (this can take a minute)...")
    state["orchestrator"] = Orchestrator()
    print("✅ Orchestrator ready — the API can now take questions.")
    yield
    state["orchestrator"] = None


app = FastAPI(title="Multi-Agent RAG Document Analyzer", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    id: int
    answer: str


class TextResponse(BaseModel):
    text: str


def get_orchestrator() -> Orchestrator:
    orchestrator = state["orchestrator"]
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Still loading models, try again in a moment.")
    return orchestrator


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question can't be empty.")

    orchestrator = get_orchestrator()
    result = orchestrator.handle_question(question)
    return ChatResponse(**result)


@app.post("/api/ocr", response_model=TextResponse)
async def ocr(image: UploadFile = File(...)):
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file.")

    try:
        text = ocr_agent.extract_text(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OCR failed: {e}")

    return TextResponse(text=text)


@app.get("/api/report/{entry_id}")
def report(entry_id: int):
    orchestrator = get_orchestrator()
    output_path = os.path.join(REPORTS_DIR, f"report_{entry_id}.pdf")

    result = orchestrator.generate_report(entry_id, output_path)
    if result is None:
        raise HTTPException(status_code=404, detail="No answer found with that id.")

    return FileResponse(output_path, media_type="application/pdf", filename=f"report_{entry_id}.pdf")


@app.post("/api/reset")
def reset_chat():
    orchestrator = get_orchestrator()
    orchestrator.chat_history = []
    return {"status": "cleared"}


@app.get("/api/health")
def health():
    return {"ready": state["orchestrator"] is not None}


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")
