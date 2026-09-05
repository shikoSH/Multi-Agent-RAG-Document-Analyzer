from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from dotenv import load_dotenv
import os


load_dotenv()


API_KEY = os.getenv("API_KEY")


# fist step define the state
class simpleState(TypedDict):
    question: str
    evidence: list  # — the chunks (with source, page, text)
    analysis: str  # — from AnalystAgent.analyze()["analysis"]
    calculations: list  # — from AnalystAgent.analyze()["calculations"]
    citations: list  # — filled in by the Citation Formatter node
    sources: list  # — filled in by the Source Formatter node
    final_answer: str  # — filled in by the Response Formatter node, this is what you return


class AnswerAgent:
    def __init__(self):
        self.client = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0,
            max_tokens=2000,
            api_key=API_KEY,
        )

        # the three formatter nodes
        def citation_formatter(State: simpleState) -> dict:
            # Format the citations
            # build one short tag per unique (source, page) pair found in the evidence
            seen = set()
            citations = []
            for chunk in State["evidence"]:
                key = (chunk["source"], chunk.get("page"))
                if key in seen:
                    continue
                seen.add(key)
                if chunk.get("page") is not None:
                    citations.append(f"{chunk['source']}, p.{chunk['page']}")
                else:
                    citations.append(f"{chunk['source']}")
            return {"citations": citations}

        def source_formatter(State: simpleState) -> dict:
            # Format the sources
            # same dedup idea as citation_formatter, but this is the end-of-answer source list
            seen = set()
            sources = []
            for chunk in State["evidence"]:
                key = (chunk["source"], chunk.get("page"))
                if key in seen:
                    continue
                seen.add(key)
                if chunk.get("page") is not None:
                    sources.append(f"{chunk['source']} (page {chunk['page']})")
                else:
                    sources.append(f"{chunk['source']}")
            return {"sources": sources}

        def response_formatter(State: simpleState) -> dict:
            # Format the final answer
            # ask the LLM to turn the analysis into a polished answer, weaving in citations
            citations_text = "\n".join(State["citations"]) if State["citations"] else "لا توجد"
            sources_text = "\n".join(State["sources"]) if State["sources"] else "لا توجد"

            prompt = f"""بناءً على السؤال والتحليل التالي، اكتب إجابة نهائية واضحة ومنظمة للمستخدم.
ضع الاستشهادات المناسبة بجانب كل ادعاء باستخدام الصيغة [المصدر]، ثم أضف قائمة المصادر في النهاية.

السؤال: {State["question"]}

التحليل:
{State["analysis"]}

الاستشهادات المتاحة:
{citations_text}

المصادر المتاحة:
{sources_text}

الإجابة النهائية:"""

            response = self.client.invoke(prompt)
            return {"final_answer": response.content.strip()}

        # step 2 create the graph
        graph = StateGraph(simpleState)

        # step 3 add the nodes
        graph.add_node("citation_formatter", citation_formatter)
        graph.add_node("source_formatter", source_formatter)
        graph.add_node("response_formatter", response_formatter)

        # step 4 add edges
        graph.add_edge(START, "citation_formatter")
        graph.add_edge("citation_formatter", "response_formatter")
        graph.add_edge("response_formatter", END)

        graph.add_edge(START, "source_formatter")
        graph.add_edge("source_formatter", "response_formatter")
        graph.add_edge("response_formatter", END)

        # step 5 compile
        self.app = graph.compile()

    def generate(self, question, analysis_result):
        initial_state = {
            "question": question,
            "evidence": analysis_result["evidence"],
            "analysis": analysis_result["analysis"],
            "calculations": analysis_result["calculations"],
            "citations": [],
            "sources": [],
            "final_answer": ""
        }
        result = self.app.invoke(initial_state)
        return result["final_answer"]


# ------------------ التشغيل والتجربة ------------------
if __name__ == "__main__":
    agent = AnswerAgent()

    # small hand-built fake analysis_result so this file can be tested on its own,
    # before the orchestrator exists to feed it real data
    fake_analysis_result = {
        "evidence": [
            {"text": "RAG allows access to external knowledge without retraining.", "source": "rag.pdf", "page": 5},
            {"text": "RAG can update knowledge dynamically and reduce hallucinations.", "source": "rag.pdf", "page": 8},
        ],
        "analysis": "RAG يوفر ميزتين أساسيتين: (1) إمكانية تحديث المعرفة بدون إعادة تدريب النموذج، (2) تقليل الهلوسة عن طريق الاعتماد على مصادر خارجية موثوقة.",
        "calculations": [],
    }

    answer = agent.generate("What are the advantages of RAG?", fake_analysis_result)
    print("\n📝 الإجابة النهائية:")
    print(answer)
