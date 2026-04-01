# ⚛️ Quantum Computing RAG System

A production-ready **Retrieval-Augmented Generation (RAG)** system built from scratch on a specialized **quantum computing dataset**, with a complete **evaluation framework** for measuring answer quality.

---

## 🚀 Overview

This project demonstrates how to build a **full AI pipeline** combining:

- 🔍 Semantic Retrieval (TF-IDF)
- 🧠 Context-Aware Generation (LLM / Gemini-ready)
- 📊 Multi-metric Evaluation System
- 💬 Chat-based Interface (Streamlit)

Unlike typical projects, this system focuses on:
> **Retrieval quality + evaluation + system design**

---

## 🎯 Key Features

- ✅ Custom RAG pipeline (no heavy frameworks)
- ✅ TF-IDF based retrieval (zero dependency option)
- ✅ Plug-and-play LLM support (Gemini / Claude / fallback)
- ✅ Evaluation using multiple metrics:
  - F1 Score (keyword overlap)
  - Cosine Similarity
  - BLEU Score
  - Retrieval Accuracy
- ✅ Fallback system (works even without API)

---

## 📂 Dataset

- Domain: **Quantum Computing**
- Size: **7 documents (~4000 words)**

Topics include:
- Origins (Feynman, Deutsch)
- Algorithms (Shor, Grover)
- Error correction
- Cryptography
- Industry & applications

📌 Includes:
- 15 QA pairs (`qa_pairs.json`)
- Key facts for evaluation

---

## 🏗️ Project Structure
rag_system/
│
├── dataset/
├── src/
│ ├── rag_pipeline.py
│ └── evaluation.py
│
├── results/
├── main.py
├── app.py # Streamlit UI
├── requirements.txt
└── README.md

---

## ⚙️ Quick Start

```bash
pip install -r requirements.txt

# Run chatbot UI
streamlit run app.py

# CLI demo
python main.py --mode demo

# Ask question
python main.py --ask "What is Grover's algorithm?"

# Run evaluation
python main.py --mode evaluate
