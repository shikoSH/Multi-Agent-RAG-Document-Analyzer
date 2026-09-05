"""
orchestrator.py
خطوة 4: Orchestrator - يربط الثلاث agents مع بعض ويدير المحادثة
Retriever Agent -> Analyst Agent -> Answer Agent
"""

from .agents.retriever_agent import RetrieverAgent
from .agents.analyst_agent import AnalystAgent
from .agents.answer_agent import AnswerAgent


class Orchestrator:
    def __init__(self):
        print("🔄 جاري تجهيز الـ Orchestrator...")
        self.retriever = RetrieverAgent()
        self.analyst = AnalystAgent(retriever_agent=self.retriever)
        self.answer_agent = AnswerAgent()

        self.chat_history = []  # list of (question, answer) tuples
        print("✅ الـ Orchestrator جاهز")

    def _format_chat_history(self) -> str:
        if not self.chat_history:
            return ""
        lines = []
        for q, a in self.chat_history:
            lines.append(f"سؤال: {q}")
            lines.append(f"إجابة: {a}")
        return "\n".join(lines)

    def handle_question(self, question: str) -> str:
        history_text = self._format_chat_history()

        evidence = self.retriever.retrieve(question, chat_history=history_text)

        if not evidence:
            answer = "❌ ما لقيت معلومات كافية بالمستندات للإجابة على سؤالك."
            self.chat_history.append((question, answer))
            return answer

        analysis_result = self.analyst.analyze(question, evidence)

        answer = self.answer_agent.generate(question, analysis_result)

        self.chat_history.append((question, answer))

        return answer


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

        answer = orchestrator.handle_question(question)
        print(f"\n🤖 الإجابة:\n{answer}\n")
