#!/usr/bin/env python3
"""
main.py — Entry point for the Quantum Computing RAG System.

Usage:
  python main.py --mode index          # Build the vector index
  python main.py --mode query          # Interactive Q&A
  python main.py --mode evaluate       # Run full evaluation
  python main.py --mode demo           # Run demo queries
  python main.py --ask "your question" # Single question
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from rag_pipeline import RAGPipeline
from evaluation import RAGEvaluator, print_evaluation_report, print_qualitative_assessment


DEMO_QUESTIONS = [
    "Who proposed the concept of quantum computing and in what year?",
    "What is Shor's algorithm and why is it significant for cryptography?",
    "What is the surface code in quantum error correction?",
    "What four post-quantum cryptography algorithms did NIST standardize in 2022?",
    "What is the difference between superconducting qubits and ion trap qubits?",
]


def build_rag(force_rebuild: bool = False) -> RAGPipeline:
    dataset_dir = str(Path(__file__).parent / "dataset")
    index_path = str(Path(__file__).parent / "results" / "vector_store.pkl")
    embedder_path = str(Path(__file__).parent / "results" / "embedder.pkl")
    os.makedirs(Path(__file__).parent / "results", exist_ok=True)

    rag = RAGPipeline(
        dataset_dir=dataset_dir,
        index_path=index_path,
        embedder_path=embedder_path,
        chunk_size=400,
        chunk_overlap=80,
        top_k=5,
    )
    rag.build_index(force_rebuild=force_rebuild)
    return rag


def mode_demo(rag: RAGPipeline):
    print("\n" + "="*65)
    print("  QUANTUM COMPUTING RAG — DEMO MODE")
    print("="*65)
    for q in DEMO_QUESTIONS:
        print(f"\n📌 Q: {q}")
        result = rag.query(q)
        print(f"💬 A: {result['answer']}")
        print(f"\n📚 Sources retrieved:")
        for r in result['retrieved_chunks'][:3]:
            print(f"   [{r.rank}] {r.chunk.doc_title} (score={r.score:.3f})")
        print("-" * 65)


def mode_interactive(rag: RAGPipeline):
    print("\n" + "="*65)
    print("  QUANTUM COMPUTING RAG — INTERACTIVE MODE")
    print("  Type your question. Commands: 'quit', 'sources'")
    print("="*65)
    show_sources = True
    while True:
        try:
            user_input = input("\n🔍 Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit', 'q'):
            break
        if user_input.lower() == 'sources':
            show_sources = not show_sources
            print(f"Source display: {'ON' if show_sources else 'OFF'}")
            continue

        result = rag.query(user_input)
        print(f"\n💬 A: {result['answer']}")

        if show_sources:
            print(f"\n📚 Sources (top 3):")
            for r in result['retrieved_chunks'][:3]:
                print(f"   [{r.rank}] {r.chunk.doc_title} | score={r.score:.3f}")
                print(f"       \"{r.chunk.text[:80]}...\"")


def mode_evaluate(rag: RAGPipeline, n_samples: int = None):
    qa_path = str(Path(__file__).parent / "dataset" / "qa_pairs.json")
    output_path = str(Path(__file__).parent / "results" / "evaluation_results.json")

    evaluator = RAGEvaluator(rag, qa_path=qa_path)

    # Load and optionally sample
    samples = evaluator.load_qa_pairs()
    if n_samples:
        samples = samples[:n_samples]
        sample_ids = [s.id for s in samples]
        results = evaluator.run_evaluation(sample_ids=sample_ids)
    else:
        results = evaluator.run_evaluation()

    agg = evaluator.aggregate_metrics(results)
    print_evaluation_report(results, agg)
    data = evaluator.save_results(output_path, results)

    print("\n\n=== QUALITATIVE ASSESSMENT INTERFACE ===")
    print_qualitative_assessment(results, n_sample=5)

    return results, agg


def mode_single_question(rag: RAGPipeline, question: str):
    print(f"\n📌 Question: {question}")
    result = rag.query(question)
    print(f"\n💬 Answer:\n{result['answer']}")
    print(f"\n📚 Retrieved Sources:")
    for r in result['retrieved_chunks']:
        print(f"   [{r.rank}] Score={r.score:.3f} | {r.chunk.doc_title}")
        print(f"       {r.chunk.text[:120]}...")


def main():
    parser = argparse.ArgumentParser(description="Quantum Computing RAG System")
    parser.add_argument("--mode", choices=["index", "query", "evaluate", "demo"],
                        default="demo", help="Operation mode")
    parser.add_argument("--ask", type=str, help="Single question to answer")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild index")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Number of QA pairs to evaluate (default: all)")
    args = parser.parse_args()

    rag = build_rag(force_rebuild=args.rebuild)

    if args.ask:
        mode_single_question(rag, args.ask)
    elif args.mode == "index":
        print("[✓] Index built successfully.")
    elif args.mode == "query":
        mode_interactive(rag)
    elif args.mode == "evaluate":
        mode_evaluate(rag, n_samples=args.n_samples)
    elif args.mode == "demo":
        mode_demo(rag)


if __name__ == "__main__":
    main()
