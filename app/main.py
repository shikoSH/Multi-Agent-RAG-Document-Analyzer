"""
main.py
FastAPI backend for the Multi-Agent RAG Document Analyzer.

Run from the project root with:
    uvicorn app.main:app --reload

It does three things:
1. Loads the Orchestrator (and therefore the embedding + reranker models)
   exactly once, when the server starts - not on every request, since
   that would be far too slow.
2. Exposes a small JSON API (/api/chat, /api/reset, /api/health) that the
   chat UI talks to.
3. Serves the chat UI itself (static/index.html) at "/".
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .orchestrator import Orchestrator

# Holds the single Orchestrator instance once it's built. A plain dict
# (rather than a global variable) so it's easy to reference from the
# lifespan function and the route handlers below.
state = {"orchestrator": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔄 Loading models and building the Orchestrator (this can take a minute)...")
    state["orchestrator"] = Orchestrator()
    print("✅ Orchestrator ready — the API can now take questions.")
    yield
    state["orchestrator"] = None


app = FastAPI(title="Multi-Agent RAG Document Analyzer", lifespan=lifespan)

# Same-origin in normal use (the UI is served by this same app), but this
# keeps things working if you ever open the UI from a different port/host
# during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


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
    answer = orchestrator.handle_question(question)
    return ChatResponse(answer=answer)


@app.post("/api/reset")
def reset_chat():
    orchestrator = get_orchestrator()
    orchestrator.chat_history = []
    return {"status": "cleared"}


@app.get("/api/health")
def health():
    return {"ready": state["orchestrator"] is not None}


# Static assets (the UI's own JS/CSS, if you split them out later).
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")
