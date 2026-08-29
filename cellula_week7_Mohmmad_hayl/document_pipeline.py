"""
document_pipeline.py
خطوة 1: تحضير المستندات - استخراج، تنظيف، تقطيع، embeddings، تخزين FAISS
"""

import os
import re
import pickle
from pathlib import Path

import fitz  # PyMuPDF
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


# ------------------ الإعدادات ------------------
DOCS_FOLDER = "./documents"          # فولدر المستندات (PDF/DOCX/TXT)
INDEX_PATH = "./vector_store.index"  # ملف FAISS
METADATA_PATH = "./metadata.pkl"     # ملف الميتاداتا (نص + مصدر + صفحة)
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


# ------------------ 1. استخراج النص ------------------
def extract_text_from_pdf(path: str):
    """يرجع قائمة (نص, رقم_الصفحة) لكل صفحة"""
    pages = []
    doc = fitz.open(path)
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append((text, i + 1))
    doc.close()
    return pages


def extract_text_from_docx(path: str):
    document = docx.Document(path)
    full_text = "\n".join(p.text for p in document.paragraphs)
    return [(full_text, None)]  # الوورد ما فيه ترقيم صفحات مباشر


def extract_text_from_txt(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [(f.read(), None)]


def extract_text(path: str):
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext == ".docx":
        return extract_text_from_docx(path)
    elif ext == ".txt":
        return extract_text_from_txt(path)
    else:
        raise ValueError(f"صيغة غير مدعومة: {ext}")


# ------------------ 2. تنظيف النص ------------------
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)          # مسافات زايدة
    text = re.sub(r"[•●▪]", "", text)         # رموز bullet
    text = text.strip()
    return text


# ------------------ 3. التقطيع (Chunking) ------------------
def chunk_documents(docs_folder: str):
    """
    يمشي على كل الملفات بالفولدر، يستخرج النص، ينظفه، يقطعه،
    ويرجع قائمة dict فيها: text, source, page
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    for filename in os.listdir(docs_folder):
        filepath = os.path.join(docs_folder, filename)
        if not os.path.isfile(filepath):
            continue

        try:
            pages = extract_text(filepath)
        except ValueError:
            print(f"⏭️  تخطي ملف غير مدعوم: {filename}")
            continue

        for text, page_num in pages:
            text = clean_text(text)
            if not text:
                continue
            chunks = splitter.split_text(text)
            for chunk in chunks:
                all_chunks.append({
                    "text": chunk,
                    "source": filename,
                    "page": page_num,
                })

    return all_chunks


# ------------------ 4. Embeddings + تخزين FAISS ------------------
def build_vector_store(chunks: list):
    model = SentenceTransformer(EMBEDDING_MODEL)

    # موديل e5 بيفضل نضيف "passage: " قبل كل نص وقت الفهرسة
    texts = ["passage: " + c["text"] for c in chunks]
    embeddings = model.encode(
        texts, show_progress_bar=True, normalize_embeddings=True
    )
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product (مع normalize = cosine similarity)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(chunks, f)

    return index, chunks


# ------------------ التشغيل ------------------
if __name__ == "__main__":
    print("📂 جاري قراءة وتقطيع المستندات...")
    chunks = chunk_documents(DOCS_FOLDER)
    print(f"✅ تم إنشاء {len(chunks)} chunk من مجلد {DOCS_FOLDER}")

    if not chunks:
        print("⚠️ ما لقيت أي مستندات. حط ملفاتك بفولدر ./documents وجرب من جديد.")
    else:
        print("🧠 جاري بناء embeddings وتخزينها بـ FAISS...")
        index, chunks = build_vector_store(chunks)
        print(f"✅ تم حفظ الـ index بـ {INDEX_PATH}")
        print(f"✅ تم حفظ الميتاداتا بـ {METADATA_PATH}")

        # عرض مثال للتحقق
        print("\n--- مثال على أول chunk ---")
        print(chunks[0])