"""
orchestrator.py
Retriever Agent -> Analyst Agent -> Answer Agent, plus the Report
Generator Agent for turning any past answer into a PDF.
"""

from .agents.retriever_agent import RetrieverAgent
from .agents.analyst_agent import AnalystAgent
from .agents.answer_agent import AnswerAgent
from .agents.report_generator_agent import ReportGeneratorAgent


class Orchestrator:
    def __init__(self):
        print("🔄 جاري تجهيز الـ Orchestrator...")
        self.retriever = RetrieverAgent()
        self.analyst = AnalystAgent(retriever_agent=self.retriever)
        self.answer_agent = AnswerAgent()
        self.report_agent = ReportGeneratorAgent()

        self.chat_history = []  # list of (question, answer) tuples - fed back into the retriever as context

        # Full record of every question answered this session, keyed by an
        # incrementing id, so the Report Generator Agent can rebuild a PDF
        # for any past answer (not just the most recent one).
        self.entries = {}
        self._next_id = 1

        print("✅ الـ Orchestrator جاهز")

    def _format_chat_history(self) -> str:
        if not self.chat_history:
            return ""
        lines = []
        for q, a in self.chat_history:
            lines.append(f"سؤال: {q}")
            lines.append(f"إجابة: {a}")
        return "\n".join(lines)

    def handle_question(self, question: str) -> dict:
        """
        Returns {"id": int, "answer": str}. The id is what the report
        endpoint uses later to regenerate this exact answer as a PDF.
        """
        history_text = self._format_chat_history()

        evidence = self.retriever.retrieve(question, chat_history=history_text)

        if not evidence:
            answer = "❌ ما لقيت معلومات كافية بالمستندات للإجابة على سؤالك."
            entry_id = self._store_entry(question, answer, analysis="", calculations=[], evidence=[])
            self.chat_history.append((question, answer))
            return {"id": entry_id, "answer": answer}

        analysis_result = self.analyst.analyze(question, evidence)
        answer = self.answer_agent.generate(question, analysis_result)

        entry_id = self._store_entry(
            question,
            answer,
            analysis=analysis_result["analysis"],
            calculations=analysis_result["calculations"],
            evidence=analysis_result["evidence"],
        )

        self.chat_history.append((question, answer))

        return {"id": entry_id, "answer": answer}

    def _store_entry(self, question, answer, analysis, calculations, evidence) -> int:
        entry_id = self._next_id
        self._next_id += 1
        self.entries[entry_id] = {
            "title": "تقرير نتائج البحث",
            "question": question,
            "final_answer": answer,
            "analysis": analysis,
            "calculations": calculations,
            "evidence": evidence,
        }
        return entry_id

    def generate_report(self, entry_id: int, output_path: str):
        """Builds a PDF for a past answer. Returns the output path, or
        None if that id doesn't exist (already answered, or never was)."""
        entry = self.entries.get(entry_id)
        if entry is None:
            return None
        return self.report_agent.generate_pdf(entry, output_path=output_path)


# ------------------ التشغيل المستقل (بدون FastAPI) ------------------
# للتجربة السريعة من التيرمنال: python -m app.orchestrator
if __name__ == "__main__":
    orchestrator = Orchestrator()

    print("\n💬 اكتب سؤالك (أو اكتب 'خروج' للإنهاء)\n")

    while True:
        question = input("أنت: ").strip()
        if question.lower() in ("خروج", "exit", "quit"):
            print("👋 مع السلامة")
            break
        if not question:
            continue

        result = orchestrator.handle_question(question)
        print(f"\n🤖 الإجابة:\n{result['answer']}\n")
