"""
analyst_agent.py
خطوة 3: Analyst Agent - يحلل الأدلة المسترجعة من Retriever Agent
يعتمد على retriever_agent.py (RetrieverAgent) للـ feedback loop
"""

import re
import json
from openai import OpenAI


# ------------------ الإعدادات ------------------
OPENROUTER_API_KEY = "api-key" # ضع مفتاح OpenRouter الحقيقي هنا (نفسه المستخدم بـ retriever_agent.py)
OPENROUTER_MODEL = "openai/gpt-4o-mini"

MAX_FEEDBACK_LOOPS = 2  # أقصى عدد مرات يرجع فيها يطلب أدلة إضافية


class AnalystAgent:
    def __init__(self, retriever_agent=None):
        """
        retriever_agent: كائن RetrieverAgent (من retriever_agent.py)
        لازم يكون موجود عشان يشتغل الـ Feedback Loop (أداة Search/Retrieve More Evidence)
        """
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        self.retriever = retriever_agent

    # ---------- استدعاء LLM مساعد ----------
    def _call_llm(self, prompt: str, temperature: float = 0.1) -> str:
        response = self.client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    # ---------- 1. Calculator ----------
    def calculate(self, expression: str) -> float:
        """
        يحسب تعبير رياضي بسيط بأمان (بدون eval مباشر على مدخلات حرة)
        مثال: "sum([85.2, 87.1, 90.3]) / 3"
        """
        allowed_names = {"sum": sum, "len": len, "max": max, "min": min, "abs": abs, "round": round}
        try:
            if not re.match(r"^[0-9\.\+\-\*\/\(\)\,\s\w\[\]]+$", expression):
                raise ValueError("تعبير غير آمن")
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return result
        except Exception as e:
            return f"خطأ بالحساب: {e}"

    # ---------- 2. Table Extractor ----------
    def extract_table(self, evidence_chunks: list) -> str:
        """
        يطلب من الـ LLM يحول أي جدول موجود بالنصوص لصيغة Markdown منظمة
        """
        combined_text = "\n\n".join(
            f"[المصدر: {c['source']}, صفحة: {c.get('page')}]\n{c['text']}"
            for c in evidence_chunks
        )
        prompt = f"""فيما يلي مقاطع نصوص من مستندات قد تحتوي على بيانات جدولية
(مثل نتائج تجارب، مقاييس أداء، أرقام مقارنة).
استخرج أي جدول أو بيانات منظمة ورتبها بصيغة Markdown Table.
إذا ما فيه بيانات جدولية واضحة، اكتب "لا توجد بيانات جدولية".

النصوص:
{combined_text}

الجدول المستخرج:"""
        return self._call_llm(prompt)

    # ---------- 3. Document Comparison ----------
    def compare_documents(self, evidence_chunks: list, comparison_aspect: str = "") -> str:
        """
        يقارن المعلومات بين مصادر مختلفة
        """
        sources = {}
        for c in evidence_chunks:
            sources.setdefault(c["source"], []).append(c["text"])

        combined = "\n\n".join(
            f"=== المصدر: {src} ===\n" + "\n".join(texts)
            for src, texts in sources.items()
        )

        aspect_note = f"ركز المقارنة على: {comparison_aspect}" if comparison_aspect else ""
        prompt = f"""قارن المعلومات التالية بين المصادر المختلفة من ناحية
التشابه والاختلاف والنتائج. {aspect_note}

{combined}

المقارنة:"""
        return self._call_llm(prompt)

    # ---------- 4. Data Analysis ----------
    def analyze_data(self, evidence_chunks: list, question: str) -> str:
        """
        تحليل نوعي/كمي عام على الأدلة بناءً على سؤال المستخدم
        """
        combined_text = "\n\n".join(
            f"[المصدر: {c['source']}, صفحة: {c.get('page')}]\n{c['text']}"
            for c in evidence_chunks
        )
        prompt = f"""حلل النصوص التالية للإجابة على السؤال، مع التركيز على أي أرقام،
اتجاهات، أو نتائج مهمة.

قواعد مهمة بخصوص الحسابات:
- إذا احتجت حساب رياضي (متوسط، فرق، نسبة مئوية، مقارنة أرقام)، اكتبه بالصيغة: CALC[التعبير]
- استخدم فقط أرقام فعلية داخل CALC[]، ولا تستخدم أبداً أسماء متغيرات أو نصوص غير رقمية
  مثال صحيح: CALC[(85.2 + 90.1) / 2]
  مثال خاطئ: CALC[len(original_string) > len(compressed_string)]
- لا تضع أي تنسيق Markdown (مثل ** أو `) حول كلمة CALC

السؤال: {question}

النصوص:
{combined_text}

التحليل:"""
        return self._call_llm(prompt)

    # ---------- 5. Search / Retrieve More Evidence (Feedback Loop) ----------
    def check_sufficiency(self, question: str, evidence_chunks: list) -> dict:
        """
        يسأل الـ LLM: هل الأدلة الحالية كافية للإجابة، أو محتاجين نطلب أكتر؟
        يرجع dict فيه: sufficient (bool), missing_info (str)
        """
        combined_text = "\n\n".join(c["text"] for c in evidence_chunks)
        prompt = f"""بناءً على السؤال والأدلة المتوفرة، حدد إذا كانت الأدلة كافية للإجابة
بشكل كامل ودقيق أم لا.

السؤال: {question}
الأدلة المتوفرة:
{combined_text}

أجب فقط بصيغة JSON بدون أي نص إضافي:
{{"sufficient": true أو false, "missing_info": "وصف قصير لأي معلومة ناقصة إن وجدت"}}"""

        raw = self._call_llm(prompt)
        try:
            cleaned = re.sub(r"```json|```", "", raw).strip()
            return json.loads(cleaned)
        except Exception:
            return {"sufficient": True, "missing_info": ""}  # افتراضياً نكمل لو فشل التحليل

    def analyze(self, question: str, evidence_chunks: list) -> dict:
        """
        الدالة الرئيسية: تدير كل أدوات التحليل + الـ Feedback Loop مع Retriever
        ترجع dict فيه: evidence (نهائي), analysis (نص التحليل), calculations (قائمة نتائج حسابية)
        """
        loops = 0
        current_evidence = evidence_chunks

        # --- Feedback Loop: تحقق من كفاية الأدلة ---
        while loops < MAX_FEEDBACK_LOOPS:
            check = self.check_sufficiency(question, current_evidence)
            if check.get("sufficient", True) or not self.retriever:
                break

            missing = check.get("missing_info", "")
            print(f"🔁 الأدلة ناقصة، جاري طلب معلومات إضافية عن: {missing}")
            extra_evidence = self.retriever.retrieve(missing or question)
            current_evidence = self._merge_unique(current_evidence, extra_evidence)
            loops += 1

        # --- التحليل الأساسي ---
        analysis_text = self.analyze_data(current_evidence, question)

        # --- تنفيذ أي حسابات مطلوبة (CALC[...]) ---
        calculations = []
        # regex مرن يتجاهل أي ** أو ` أو مسافات حوالين CALC
        for match in re.finditer(r"\**`?CALC`?\**\s*\[(.*?)\]", analysis_text):
            expr = match.group(1)
            result = self.calculate(expr)
            calculations.append({"expression": expr, "result": result})
            analysis_text = analysis_text.replace(match.group(0), f"({result})")

        return {
            "evidence": current_evidence,
            "analysis": analysis_text,
            "calculations": calculations,
        }

    @staticmethod
    def _merge_unique(list1: list, list2: list) -> list:
        seen = {(c["text"][:80], c["source"]) for c in list1}
        merged = list(list1)
        for c in list2:
            key = (c["text"][:80], c["source"])
            if key not in seen:
                merged.append(c)
                seen.add(key)
        return merged


# ------------------ التشغيل / التجربة ------------------
if __name__ == "__main__":
    from retriever_agent import RetrieverAgent

    retriever = RetrieverAgent()
    analyst = AnalystAgent(retriever_agent=retriever)

    question = input("اكتب سؤالك: ")
    evidence = retriever.retrieve(question)

    result = analyst.analyze(question, evidence)

    print("\n📊 التحليل:")
    print(result["analysis"])

    if result["calculations"]:
        print("\n🧮 الحسابات المنفذة:")
        for calc in result["calculations"]:
            print(f"  {calc['expression']} = {calc['result']}")

    print(f"\n📚 عدد الأدلة النهائية المستخدمة: {len(result['evidence'])}")