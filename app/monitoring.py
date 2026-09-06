"""
monitoring.py
Turns on LangSmith tracing for the whole app if LANGSMITH_API_KEY is set
in .env. Call setup_langsmith() once, before any agent is created.

- LangChain/LangGraph calls (ChatOpenAI in answer_agent.py, the
  StateGraph it runs in) are traced automatically once these env vars
  are set - no extra code needed there.
- Plain OpenAI client calls (used by the Retriever and Analyst agents,
  which don't go through LangChain) are traced by wrapping the client
  with `wrap_openai` - see agents/retriever_agent.py and
  agents/analyst_agent.py.
"""
import os
from dotenv import load_dotenv

load_dotenv()


def setup_langsmith():
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        print("ℹ️  LANGSMITH_API_KEY not set - tracing is off.")
        return

    project = os.getenv("LANGSMITH_PROJECT", "multi-agent-rag-system")

    # both the new (LANGSMITH_*) and legacy (LANGCHAIN_*) names are set,
    # since different library versions still read the old ones.
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGSMITH_PROJECT"] = project
    os.environ["LANGCHAIN_PROJECT"] = project

    print(f"✅ LangSmith tracing on — project: {project}")
