# ============================================================
# FINAL RAG PIPELINE (GEMINI + FULLY FIXED)
# ============================================================

import os
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field

# ✅ NEW GEMINI SDK
from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class Document:
    doc_id: str
    title: str
    content: str
    filepath: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    doc_title: str = ""   # ✅ FIXED
    embedding: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    rank: int


# ============================================================
# DOCUMENT LOADER
# ============================================================

class DocumentLoader:
    def load_from_directory(self, directory: str) -> List[Document]:
        docs = []
        for filepath in sorted(Path(directory).glob("*.txt")):
            content = filepath.read_text(encoding="utf-8")
            lines = content.strip().split("\n")
            title = lines[0].replace("Title:", "").strip() if lines[0].startswith("Title:") else filepath.stem

            docs.append(Document(
                doc_id=filepath.stem,
                title=title,
                content=content,
                filepath=str(filepath)
            ))

        print(f"[Loader] Loaded {len(docs)} documents.")
        return docs


# ============================================================
# CHUNKER
# ============================================================

class TextChunker:
    def __init__(self, chunk_size=400, overlap=80):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        chunks = []

        for doc in documents:
            text = doc.content
            paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]

            current = ""
            idx = 0

            for para in paragraphs:
                if len(current + para) <= self.chunk_size:
                    current += "\n\n" + para
                else:
                    if current:
                        chunks.append(Chunk(
                            chunk_id=f"{doc.doc_id}_{idx}",
                            doc_id=doc.doc_id,
                            text=current,
                            doc_title=doc.title   # ✅ FIXED
                        ))
                        idx += 1
                    current = para

            if current:
                chunks.append(Chunk(
                    chunk_id=f"{doc.doc_id}_{idx}",
                    doc_id=doc.doc_id,
                    text=current,
                    doc_title=doc.title   # ✅ FIXED
                ))

        print(f"[Chunker] Created {len(chunks)} chunks.")
        return chunks


# ============================================================
# TF-IDF EMBEDDER (FREE)
# ============================================================

class TFIDFEmbedder:
    def __init__(self):
        self.vocab = {}
        self.idf = {}

    def fit(self, texts: List[str]):
        df = {}
        for text in texts:
            tokens = set(self.tokenize(text))
            for t in tokens:
                df[t] = df.get(t, 0) + 1

        self.vocab = {t: i for i, t in enumerate(df.keys())}
        N = len(texts)
        self.idf = {t: np.log((N + 1) / (df[t] + 1)) + 1 for t in df}

    def transform(self, text: str):
        vec = np.zeros(len(self.vocab))
        tokens = self.tokenize(text)

        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1

        for t, count in tf.items():
            if t in self.vocab:
                vec[self.vocab[t]] = count * self.idf.get(t, 1)

        norm = np.linalg.norm(vec)
        return vec / norm if norm != 0 else vec

    def tokenize(self, text):
        return re.findall(r'\b\w+\b', text.lower())


# ============================================================
# VECTOR STORE
# ============================================================

class VectorStore:
    def __init__(self):
        self.chunks = []
        self.embeddings = None

    def build(self, chunks: List[Chunk], embedder: TFIDFEmbedder):
        self.chunks = chunks
        self.embeddings = np.array([embedder.transform(c.text) for c in chunks])

    def search(self, query_vec, top_k=5):
        scores = self.embeddings @ query_vec
        idxs = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, i in enumerate(idxs, 1):
            results.append(RetrievalResult(
                chunk=self.chunks[i],
                score=float(scores[i]),
                rank=rank
            ))
        return results


# ============================================================
# GEMINI GENERATION (SAFE)
# ============================================================

def generate_answer(prompt: str) -> str:
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",   # ✅ FIXED MODEL
            contents=prompt
        )

        # ✅ SAFE HANDLING
        if hasattr(response, "text") and response.text:
            return response.text.strip()

        return "No response generated."

    except Exception as e:
        return f"Error generating response: {str(e)}"


# ============================================================
# MAIN PIPELINE (COMPATIBLE WITH main.py)
# ============================================================

class RAGPipeline:
    def __init__(
        self,
        dataset_dir,
        index_path=None,
        embedder_path=None,
        chunk_size=400,
        chunk_overlap=80,
        top_k=5
    ):
        self.dataset_dir = dataset_dir
        self.top_k = top_k

        self.loader = DocumentLoader()
        self.chunker = TextChunker(chunk_size, chunk_overlap)
        self.embedder = TFIDFEmbedder()
        self.vectorstore = VectorStore()

        self._build()

    def build_index(self, force_rebuild=False):
        pass  # compatibility with main.py

    def _build(self):
        docs = self.loader.load_from_directory(self.dataset_dir)
        chunks = self.chunker.chunk_documents(docs)

        texts = [c.text for c in chunks]
        self.embedder.fit(texts)
        self.vectorstore.build(chunks, self.embedder)

    def query(self, question: str):
        query_vec = self.embedder.transform(question)
        results = self.vectorstore.search(query_vec, self.top_k)

        context = "\n\n".join([r.chunk.text for r in results])

        prompt = f"""
You are a quantum computing expert.

Answer ONLY using the context below.
Do NOT use outside knowledge.

Context:
{context}

Question:
{question}

Answer:
"""

        answer = generate_answer(prompt)

        return {
            "answer": answer,
            "retrieved_chunks": results
        }