"""
retriever_agent.py
خطوة 2: Retriever Agent - يبحث بالـ Vector DB ويرجع أفضل الأدلة
تحديث: استخدام OpenRouter API بدلاً من Ollama المحلي
"""

import pickle
import numpy as np
import faiss
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi


# ------------------ الإعدادات ------------------
INDEX_PATH = "./vector_store.index"
METADATA_PATH = "./metadata.pkl"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# إعدادات OpenRouter
OPENROUTER_API_KEY = "api-key"  # ضع مفتاح OpenRouter الخاص بك هنا
OPENROUTER_MODEL = "openai/gpt-4o-mini"              # أو يمكنك استخدام "meta-llama/llama-3.1-8b-instruct"

TOP_K_INITIAL = 20     # عدد النتائج من البحث الأولي
TOP_K_FINAL = 6        # عدد النتائج بعد الـ Reranking


# ------------------ الكلاس الرئيسي ------------------
class RetrieverAgent:
    def __init__(self):
        print("🔄 جاري تحميل الـ index والموديلات...")
        self.index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            self.chunks = pickle.load(f)  # list of dict: text, source, page

        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)
        self.reranker = CrossEncoder(RERANKER_MODEL)

        # تجهيز عميل OpenRouter
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )

        # تجهيز BM25 للـ Keyword Search
        tokenized_corpus = [c["text"].split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"✅ تم تحميل {len(self.chunks)} chunk بنجاح")

    # ---------- 1. Query Rewriter (عبر OpenRouter) ----------
    def rewrite_query(self, query: str, chat_history: str = "") -> str:
        prompt = f"""أعد صياغة السؤال التالي ليكون واضحاً ومستقلاً بذاته للبحث،
مع الأخذ بعين الاعتبار سياق المحادثة السابقة إن وجد.
أعطني السؤال المعاد صياغته فقط بدون أي شرح إضافي.

سياق المحادثة: {chat_history if chat_history else "لا يوجد"}
السؤال الأصلي: {query}

السؤال المعاد صياغته:"""

        try:
            response = self.client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            rewritten = response.choices[0].message.content.strip()
            return rewritten if rewritten else query
        except Exception as e:
            print(f"⚠️ Query Rewriter فشل ({e})، رح نستخدم السؤال الأصلي")
            return query

    # ---------- 2. Semantic Search ----------
    def semantic_search(self, query: str, top_k: int = TOP_K_INITIAL):
        query_vec = self.embed_model.encode(
            ["query: " + query], normalize_embeddings=True
        )
        query_vec = np.array(query_vec, dtype="float32")
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx].copy()
            chunk["semantic_score"] = float(score)
            results.append(chunk)
        return results

    # ---------- 3. Keyword Search (BM25) ----------
    def keyword_search(self, query: str, top_k: int = TOP_K_INITIAL):
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            chunk = self.chunks[idx].copy()
            chunk["keyword_score"] = float(scores[idx])
            results.append(chunk)
        return results

    # ---------- 4. Metadata Filter ----------
    def apply_metadata_filter(self, results: list, source: str = None, page: int = None):
        if source:
            results = [r for r in results if r["source"] == source]
        if page:
            results = [r for r in results if r.get("page") == page]
        return results

    # ---------- 5. دمج Semantic + Keyword (Hybrid) ----------
    def hybrid_merge(self, semantic_results: list, keyword_results: list):
        merged = {}
        for r in semantic_results:
            key = (r["text"], r["source"])
            merged[key] = r
        for r in keyword_results:
            key = (r["text"], r["source"])
            if key in merged:
                merged[key]["keyword_score"] = r["keyword_score"]
            else:
                merged[key] = r
        return list(merged.values())

    # ---------- 6. Reranker ----------
    def rerank(self, query: str, candidates: list, top_k: int = TOP_K_FINAL):
        if not candidates:
            return []
        pairs = [[query, c["text"]] for c in candidates]
        scores = self.reranker.predict(pairs)

        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)

        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_k]

    # ---------- 7. Context Selector ----------
    def select_context(self, ranked_chunks: list, max_chunks: int = TOP_K_FINAL):
        selected = []
        seen_texts = set()
        for chunk in ranked_chunks:
            snippet = chunk["text"][:80]
            if snippet in seen_texts:
                continue
            seen_texts.add(snippet)
            selected.append(chunk)
            if len(selected) >= max_chunks:
                break
        return selected

    # ---------- الدالة الرئيسية ----------
    def retrieve(self, query: str, chat_history: str = "",
                 source_filter: str = None, page_filter: int = None):
        rewritten_query = self.rewrite_query(query, chat_history)
        print(f"📝 السؤال المعاد صياغته: {rewritten_query}")

        semantic_results = self.semantic_search(rewritten_query)
        keyword_results = self.keyword_search(rewritten_query)

        merged = self.hybrid_merge(semantic_results, keyword_results)
        merged = self.apply_metadata_filter(merged, source_filter, page_filter)

        reranked = self.rerank(rewritten_query, merged)
        final_evidence = self.select_context(reranked)

        return final_evidence


# ------------------ التشغيل والتجربة ------------------
if __name__ == "__main__":
    agent = RetrieverAgent()

    test_query = input("اكتب سؤالك: ")
    evidence = agent.retrieve(test_query)

    print(f"\n📚 تم إيجاد {len(evidence)} مقطع مرتبط:\n")
    for i, chunk in enumerate(evidence, 1):
        print(f"--- مقطع {i} (المصدر: {chunk['source']}, صفحة: {chunk.get('page')}) ---")
        print(f"درجة الترتيب: {chunk.get('rerank_score', 0):.3f}")
        print(chunk["text"][:200] + "...\n")