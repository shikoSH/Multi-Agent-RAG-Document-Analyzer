 Document Analyzer — FastAPI + UI

Retriever → Analyst → Answer pipeline, wrapped in a FastAPI backend with
a chat UI. Also supports voice questions, photo-to-text (OCR), PDF
report generation, and LangSmith tracing.

## Layout

```
app/
  agents/
    retriever_agent.py         (LangSmith-wrapped OpenAI client, via OpenRouter)
    analyst_agent.py            (LangSmith-wrapped OpenAI client, via OpenRouter)
    answer_agent.py               (also via OpenRouter now - no OpenAI key needed)
    report_generator_agent.py       (teammate's PDF report agent)
    ocr_agent.py                      (new — Tesseract OCR)
  orchestrator.py                        (wires all agents together)
  monitoring.py                            (turns on LangSmith tracing)
  main.py                                    (FastAPI app + all routes)
static/
  index.html                                   (chat UI: text/voice/image)
data/
  vector_store.index
  metadata.pkl
reports/                                          (generated PDFs land here)
```

## Setup

```powershell
py -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt

copy .env.example .env
```

Then open `.env` and fill in:
- `OPENROUTER_API_KEY` — the only LLM key you need; used by all three
  agents (Retriever, Analyst, Answer)
- `LANGSMITH_API_KEY` — optional, leave blank to run without tracing
- `ARABIC_FONT_PATH` / `ARABIC_BOLD_FONT_PATH` — only if you're not on
  Windows, or want a different font than Tahoma

No OpenAI key is needed anywhere in this project.

### Installing Tesseract (needed for the image/OCR feature)

`pytesseract` is just a Python wrapper — it calls a separate program
called Tesseract, which has to be installed on your machine:

- **Windows:** download and run the installer from
  [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki).
  During install, make sure the **Arabic** language pack is checked (it's
  in the list of additional language data).
- After installing, if `tesseract` isn't automatically on your PATH, set
  `TESSERACT_CMD` in `.env` to the full path of `tesseract.exe`
  (commonly `C:/Program Files/Tesseract-OCR/tesseract.exe`).

If you skip this, every other feature still works — you'll just get an
error if you try to upload an image.

## Run

From the project root (`data/`, `static/`, and `reports/` are all
relative paths):

```powershell
uvicorn app.main:app --reload
```

Wait for `✅ Orchestrator ready`, then open http://127.0.0.1:8000.

## API

| Endpoint | Method | What it does |
|---|---|---|
| `/api/chat` | POST | `{"question": "..."}` → `{"id": int, "answer": "..."}` |
| `/api/ocr` | POST | multipart `image` file → `{"text": "..."}` |
| `/api/report/{id}` | GET | builds and downloads a PDF for that answer |
| `/api/reset` | POST | clears chat history |
| `/api/health` | GET | `{"ready": true|false}` |

The `id` returned by `/api/chat` is what the "Download PDF report"
button under each answer in the UI uses to call `/api/report/{id}`.

## How the new features fit in

- **Voice query** — the mic button uses the browser's own Web Speech API
  (`SpeechRecognition`) — free, no API key, no server call at all. Works
  in Chrome/Edge; not supported in Firefox (the button disables itself
  there). Recognized text lands in the question box for you to review
  before sending — it never auto-sends. Default language is `ar-EG`;
  change it in `static/index.html` (`recognition.lang`) if you mostly
  ask in English.
- **Image OCR** — the 📎 button uploads a photo to `/api/ocr`
  (Tesseract, `ara+eng` by default), same review-before-sending flow.
- **PDF reports** — every answered question is kept in
  `Orchestrator.entries` (question, analysis, calculations, evidence,
  final answer) so a report can be regenerated for any past answer, not
  just the latest one. `ReportGeneratorAgent` owns all the visual
  choices — fonts and colors — via its `DesignSystem` class.
- **LangSmith monitoring** — `monitoring.setup_langsmith()` runs once at
  startup. If `LANGSMITH_API_KEY` is set, it turns on tracing for
  everything: the Answer Agent's LangGraph runs are traced automatically
  once the env vars are set, and the Retriever/Analyst Agents (which use
  the plain OpenAI client, not LangChain) are traced by wrapping their
  client with `wrap_openai` — a one-line change in each file.

## Things worth knowing

- The Orchestrator is a single instance shared by every request, so chat
  history is global, not per-browser-tab. Fine for a single-user demo.
- `answer_agent.py`'s graph wires `citation_formatter` and
  `source_formatter` as two separate branches that both feed into
  `response_formatter`. Worth double-checking that this doesn't run
  `response_formatter` twice in your LangGraph version — untouched since
  it's your agent logic, just flagging it.
