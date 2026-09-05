# Document Analyzer — FastAPI + UI

Wraps your existing Retriever → Analyst → Answer pipeline in a FastAPI
backend with a chat UI.

## Layout

```
app/
  agents/
    retriever_agent.py   (unchanged logic, config now via .env)
    analyst_agent.py      (unchanged logic, config now via .env)
    answer_agent.py        (unchanged)
  orchestrator.py           (unchanged, just relative imports)
  main.py                     (new — FastAPI app)
static/
  index.html                  (new — the chat UI)
data/
  vector_store.index
  metadata.pkl
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # then fill in API_KEY and OPENROUTER_API_KEY
```

## Run

From the project root (important — `data/` and `static/` are referenced
as relative paths):

```bash
uvicorn app.main:app --reload
```

First request after startup will be slow — that's when the embedding
model and reranker load into memory. Open http://127.0.0.1:8000 once the
terminal prints "Orchestrator ready".

## API

- `POST /api/chat` — `{"question": "..."}` → `{"answer": "..."}`
- `POST /api/reset` — clears the server-side chat history
- `GET /api/health` — `{"ready": true|false}`

## Notes / things worth knowing

- The Orchestrator is a single instance shared by every request, so chat
  history is global, not per-browser-tab. Fine for a single-user demo;
  if you need multiple concurrent conversations later, that's the piece
  to change (a dict of `Orchestrator` keyed by session id).
- `answer_agent.py`'s graph wires `citation_formatter` and
  `source_formatter` as two separate branches that both feed into
  `response_formatter`. Worth double-checking that this doesn't run
  `response_formatter` twice in your LangGraph version — didn't touch
  it since it's your agent logic, just flagging it.
