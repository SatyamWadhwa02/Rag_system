"""
evaluation.py
RAG Evaluation Framework for the Quantum Computing domain.

Metrics:
  1. Keyword Overlap Score (KOS)   — precision/recall/F1 over key facts
  2. Cosine Similarity Score (CSS) — TF-IDF embedding similarity
  3. Retrieval Relevance Score (RRS) — checks if source doc is retrieved
  4. BLEU-1 Unigram Score          — surface-level n-gram overlap
  5. Qualitative Rubric Display    — human scoring interface
"""

import re
import json
import math
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path


# ──────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────

@dataclass
class EvalSample:
    """One QA pair for evaluation."""
    id: str
    question: str
    expected_answer: str
    source_doc: str
    key_facts: List[str]


@dataclass
class EvalResult:
    """Evaluation result for one QA pair."""
    sample_id: str
    question: str
    expected_answer: str
    generated_answer: str
    source_doc: str

    # Quantitative metrics
    keyword_precision: float = 0.0
    keyword_recall: float = 0.0
    keyword_f1: float = 0.0
    cosine_similarity: float = 0.0
    bleu1_score: float = 0.0
    retrieval_hit: bool = False
    retrieval_rank: Optional[int] = None

    # Composite score
    composite_score: float = 0.0

    # Retrieved docs
    retrieved_docs: List[str] = field(default_factory=list)
    retrieved_scores: List[float] = field(default_factory=list)

    # Latency
    latency_ms: float = 0.0


# ──────────────────────────────────────────────
# Metric Implementations
# ──────────────────────────────────────────────

class MetricCalculator:
    """All quantitative metric computations."""

    STOPWORDS = {
        'the','a','an','and','or','but','in','on','at','to','for','of','with',
        'is','are','was','were','be','been','being','have','has','had','do',
        'does','did','will','would','could','should','may','might','this','that',
        'it','its','they','their','by','from','as','if','so','not','also',
        'such','which','who','what','when','where','how','than','then',
    }

    def tokenize(self, text: str, remove_stopwords: bool = True) -> List[str]:
        tokens = re.findall(r'\b[a-z0-9][a-z0-9\-]{1,}\b', text.lower())
        if remove_stopwords:
            tokens = [t for t in tokens if t not in self.STOPWORDS]
        return tokens

    # ── Metric 1: Keyword Overlap Score ──────────────────────────────────────

    def keyword_overlap(self, generated: str, key_facts: List[str]) -> Tuple[float, float, float]:
        """
        Compute keyword overlap between generated answer and expected key facts.

        Each key_fact is a short phrase. A fact is "found" if all its significant
        tokens appear in the generated answer (partial credit for partial matches).

        Returns: (precision, recall, F1)
        """
        gen_tokens = set(self.tokenize(generated))
        gen_text_lower = generated.lower()

        hits = 0
        for fact in key_facts:
            fact_lower = fact.lower()
            # Exact phrase match first
            if fact_lower in gen_text_lower:
                hits += 1
                continue
            # Token-level match: all tokens of fact must appear in generated
            fact_tokens = set(self.tokenize(fact, remove_stopwords=False))
            fact_tokens_clean = set(self.tokenize(fact))
            if fact_tokens_clean and fact_tokens_clean.issubset(gen_tokens):
                hits += 1
                continue
            # Partial credit: ≥50% of fact tokens found
            if fact_tokens_clean:
                overlap_ratio = len(fact_tokens_clean & gen_tokens) / len(fact_tokens_clean)
                if overlap_ratio >= 0.5:
                    hits += 0.5

        recall = hits / len(key_facts) if key_facts else 0.0

        # Precision: how many generated words relate to expected facts
        all_fact_tokens = set()
        for fact in key_facts:
            all_fact_tokens.update(self.tokenize(fact))
        gen_relevant = len(gen_tokens & all_fact_tokens)
        precision = gen_relevant / len(gen_tokens) if gen_tokens else 0.0

        # Cap precision at 1.0 (can exceed due to counting method)
        precision = min(precision, 1.0)

        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return round(precision, 4), round(recall, 4), round(f1, 4)

    # ── Metric 2: TF-IDF Cosine Similarity ───────────────────────────────────

    def _build_tfidf_vector(self, text: str, vocabulary: Dict[str, int], idf: Dict[str, float]) -> List[float]:
        tokens = self.tokenize(text)
        tf: Dict[str, float] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        total = sum(tf.values()) or 1
        vec = [0.0] * len(vocabulary)
        for token, count in tf.items():
            if token in vocabulary:
                vec[vocabulary[token]] = (count / total) * idf.get(token, 1.0)
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def cosine_similarity_tfidf(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts using TF-IDF vectors."""
        # Build vocabulary from both texts
        tokens1 = self.tokenize(text1)
        tokens2 = self.tokenize(text2)
        all_tokens = list(set(tokens1 + tokens2))
        vocabulary = {t: i for i, t in enumerate(all_tokens)}

        # Simple IDF from the two texts
        df = {}
        for t in set(tokens1):
            df[t] = df.get(t, 0) + 1
        for t in set(tokens2):
            df[t] = df.get(t, 0) + 1
        idf = {t: math.log(3 / (freq + 1)) + 1.0 for t, freq in df.items()}

        v1 = self._build_tfidf_vector(text1, vocabulary, idf)
        v2 = self._build_tfidf_vector(text2, vocabulary, idf)

        dot = sum(a * b for a, b in zip(v1, v2))
        return round(dot, 4)

    # ── Metric 3: BLEU-1 (Unigram Precision) ─────────────────────────────────

    def bleu1(self, generated: str, reference: str) -> float:
        """
        Compute unigram BLEU score with brevity penalty.
        BLEU-1 measures what fraction of generated tokens appear in the reference.
        """
        gen_tokens = self.tokenize(generated, remove_stopwords=False)
        ref_tokens = self.tokenize(reference, remove_stopwords=False)

        if not gen_tokens:
            return 0.0

        ref_counts: Dict[str, int] = {}
        for t in ref_tokens:
            ref_counts[t] = ref_counts.get(t, 0) + 1

        gen_counts: Dict[str, int] = {}
        for t in gen_tokens:
            gen_counts[t] = gen_counts.get(t, 0) + 1

        clipped_count = 0
        for token, count in gen_counts.items():
            clipped_count += min(count, ref_counts.get(token, 0))

        precision = clipped_count / len(gen_tokens)

        # Brevity penalty
        bp = min(1.0, math.exp(1 - len(ref_tokens) / len(gen_tokens))) if len(gen_tokens) > len(ref_tokens) else 1.0

        return round(bp * precision, 4)

    # ── Metric 4: Composite Score ─────────────────────────────────────────────

    def composite_score(
        self,
        keyword_f1: float,
        cosine_sim: float,
        bleu1: float,
        retrieval_hit: bool,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Weighted composite of all metrics.
        Default weights emphasize keyword recall (most domain-specific).
        """
        if weights is None:
            weights = {
                "keyword_f1": 0.40,
                "cosine_sim": 0.25,
                "bleu1": 0.20,
                "retrieval": 0.15,
            }
        score = (
            weights["keyword_f1"] * keyword_f1 +
            weights["cosine_sim"] * cosine_sim +
            weights["bleu1"] * bleu1 +
            weights["retrieval"] * (1.0 if retrieval_hit else 0.0)
        )
        return round(score, 4)


# ──────────────────────────────────────────────
# Retrieval Evaluator
# ──────────────────────────────────────────────

class RetrievalEvaluator:
    """Evaluates whether the correct source document was retrieved."""

    def check_retrieval(
        self,
        source_doc: str,
        retrieved_chunks
    ) -> Tuple[bool, Optional[int]]:
        """
        Check if any retrieved chunk comes from the correct source document.
        Returns (hit, rank).
        """
        for result in retrieved_chunks:
            if source_doc in result.chunk.doc_id:
                return True, result.rank
        return False, None


# ──────────────────────────────────────────────
# Main Evaluator
# ──────────────────────────────────────────────

class RAGEvaluator:
    """
    Orchestrates evaluation of the RAG pipeline over all QA pairs.
    """

    def __init__(self, rag_pipeline, qa_path: str):
        self.rag = rag_pipeline
        self.qa_path = qa_path
        self.metrics = MetricCalculator()
        self.retrieval_eval = RetrievalEvaluator()
        self.results: List[EvalResult] = []

    def load_qa_pairs(self) -> List[EvalSample]:
        with open(self.qa_path) as f:
            data = json.load(f)
        samples = []
        for item in data:
            samples.append(EvalSample(
                id=item["id"],
                question=item["question"],
                expected_answer=item["expected_answer"],
                source_doc=item["source_doc"],
                key_facts=item["key_facts"]
            ))
        return samples

    def evaluate_sample(self, sample: EvalSample) -> EvalResult:
        """Run RAG and evaluate a single QA pair."""
        t0 = time.time()

        # Run retrieval + generation
        rag_output = self.rag.query(sample.question)
        latency = (time.time() - t0) * 1000

        generated = rag_output["answer"]
        retrieved = rag_output["retrieved_chunks"]

        # Compute metrics
        prec, recall, f1 = self.metrics.keyword_overlap(generated, sample.key_facts)
        cosine_sim = self.metrics.cosine_similarity_tfidf(generated, sample.expected_answer)
        bleu = self.metrics.bleu1(generated, sample.expected_answer)
        hit, rank = self.retrieval_eval.check_retrieval(sample.source_doc, retrieved)
        composite = self.metrics.composite_score(f1, cosine_sim, bleu, hit)

        result = EvalResult(
            sample_id=sample.id,
            question=sample.question,
            expected_answer=sample.expected_answer,
            generated_answer=generated,
            source_doc=sample.source_doc,
            keyword_precision=prec,
            keyword_recall=recall,
            keyword_f1=f1,
            cosine_similarity=cosine_sim,
            bleu1_score=bleu,
            retrieval_hit=hit,
            retrieval_rank=rank,
            composite_score=composite,
            retrieved_docs=[r.chunk.doc_id for r in retrieved],
            retrieved_scores=[r.score for r in retrieved],
            latency_ms=round(latency, 1)
        )
        return result

    def run_evaluation(self, sample_ids: Optional[List[str]] = None) -> List[EvalResult]:
        """Evaluate all (or selected) QA pairs."""
        samples = self.load_qa_pairs()
        if sample_ids:
            samples = [s for s in samples if s.id in sample_ids]

        print(f"\n{'='*60}")
        print(f"Running evaluation on {len(samples)} QA pairs...")
        print(f"{'='*60}\n")

        results = []
        for i, sample in enumerate(samples, 1):
            print(f"[{i}/{len(samples)}] Evaluating: {sample.id} — {sample.question[:50]}...")
            result = self.evaluate_sample(sample)
            results.append(result)
            print(f"         KF1={result.keyword_f1:.3f} | "
                  f"CSS={result.cosine_similarity:.3f} | "
                  f"BLEU={result.bleu1_score:.3f} | "
                  f"Ret={'✓' if result.retrieval_hit else '✗'} | "
                  f"Comp={result.composite_score:.3f} | "
                  f"{result.latency_ms:.0f}ms")

        self.results = results
        return results

    def aggregate_metrics(self, results: Optional[List[EvalResult]] = None) -> Dict:
        """Compute aggregate statistics over all results."""
        if results is None:
            results = self.results
        if not results:
            return {}

        n = len(results)
        def avg(vals): return round(sum(vals) / n, 4)
        def std(vals):
            m = sum(vals) / n
            return round(math.sqrt(sum((v - m) ** 2 for v in vals) / n), 4)

        kf1s = [r.keyword_f1 for r in results]
        css = [r.cosine_similarity for r in results]
        bleus = [r.bleu1_score for r in results]
        comps = [r.composite_score for r in results]
        hits = [r.retrieval_hit for r in results]
        latencies = [r.latency_ms for r in results]

        return {
            "n_samples": n,
            "keyword_f1": {"mean": avg(kf1s), "std": std(kf1s), "min": round(min(kf1s), 4), "max": round(max(kf1s), 4)},
            "cosine_similarity": {"mean": avg(css), "std": std(css)},
            "bleu1": {"mean": avg(bleus), "std": std(bleus)},
            "retrieval_accuracy": round(sum(hits) / n, 4),
            "composite_score": {"mean": avg(comps), "std": std(comps)},
            "latency_ms": {"mean": avg(latencies), "std": std(latencies)},
        }

    def save_results(self, output_path: str, results: Optional[List[EvalResult]] = None):
        """Save evaluation results to JSON."""
        if results is None:
            results = self.results
        data = {
            "aggregate": self.aggregate_metrics(results),
            "per_sample": [
                {
                    "id": r.sample_id,
                    "question": r.question,
                    "expected_answer": r.expected_answer,
                    "generated_answer": r.generated_answer,
                    "metrics": {
                        "keyword_precision": r.keyword_precision,
                        "keyword_recall": r.keyword_recall,
                        "keyword_f1": r.keyword_f1,
                        "cosine_similarity": r.cosine_similarity,
                        "bleu1": r.bleu1_score,
                        "retrieval_hit": r.retrieval_hit,
                        "retrieval_rank": r.retrieval_rank,
                        "composite_score": r.composite_score,
                        "latency_ms": r.latency_ms,
                    },
                    "retrieved_docs": r.retrieved_docs,
                }
                for r in results
            ]
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\n[Eval] Results saved to {output_path}")
        return data


# ──────────────────────────────────────────────
# Qualitative Rubric Printer
# ──────────────────────────────────────────────

QUALITATIVE_RUBRIC = """
╔══════════════════════════════════════════════════════════════════════╗
║             QUALITATIVE ASSESSMENT RUBRIC                            ║
╠══════════════════════════════════════════════════════════════════════╣
║  Score answers on each dimension: 1 (Poor) → 5 (Excellent)          ║
║                                                                      ║
║  1. FACTUAL ACCURACY                                                 ║
║     5 - All facts correct, all key details present                   ║
║     3 - Mostly correct with minor errors or omissions                ║
║     1 - Contains significant factual errors                          ║
║                                                                      ║
║  2. COMPLETENESS                                                     ║
║     5 - All expected key facts covered                               ║
║     3 - Most facts covered, minor gaps                               ║
║     1 - Major information missing                                    ║
║                                                                      ║
║  3. COHERENCE & CLARITY                                              ║
║     5 - Well-structured, clear, readable                             ║
║     3 - Understandable but somewhat disjointed                       ║
║     1 - Confusing, contradictory, or incoherent                      ║
║                                                                      ║
║  4. HALLUCINATION (ABSENCE OF)                                       ║
║     5 - No information not found in source docs                      ║
║     3 - Minor unsupported embellishments                             ║
║     1 - Contains clearly fabricated information                      ║
║                                                                      ║
║  Composite Qualitative Score = mean of all four dimensions           ║
╚══════════════════════════════════════════════════════════════════════╝
"""


def print_qualitative_assessment(results: List[EvalResult], n_sample: int = 5):
    """Print a human-readable qualitative assessment interface."""
    print(QUALITATIVE_RUBRIC)
    print(f"\nShowing {n_sample} sample answers for human evaluation:\n")

    for result in results[:n_sample]:
        print("=" * 70)
        print(f"ID: {result.sample_id}")
        print(f"Q:  {result.question}")
        print(f"\nEXPECTED ANSWER:\n{result.expected_answer}")
        print(f"\nGENERATED ANSWER:\n{result.generated_answer}")
        print(f"\nQuantitative Scores:")
        print(f"  Keyword F1:        {result.keyword_f1:.3f}")
        print(f"  Cosine Similarity: {result.cosine_similarity:.3f}")
        print(f"  BLEU-1:            {result.bleu1_score:.3f}")
        print(f"  Retrieval Hit:     {'✓' if result.retrieval_hit else '✗'} (rank {result.retrieval_rank})")
        print(f"  Composite Score:   {result.composite_score:.3f}")
        print(f"\nRetrieved from: {', '.join(result.retrieved_docs[:3])}")
        print("\n>>> HUMAN EVALUATION (1-5 for each):")
        print("    Factual Accuracy: ___  Completeness: ___  Coherence: ___  No Hallucination: ___")
        print()


# ──────────────────────────────────────────────
# Report Printer
# ──────────────────────────────────────────────

def print_evaluation_report(results: List[EvalResult], agg: Dict):
    """Print a formatted evaluation report."""
    print("\n" + "="*70)
    print("  QUANTUM COMPUTING RAG — EVALUATION REPORT")
    print("="*70)

    print(f"\n📊 AGGREGATE METRICS (n={agg['n_samples']})")
    print(f"  {'Metric':<30} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*62}")

    kf1 = agg['keyword_f1']
    css = agg['cosine_similarity']
    bl = agg['bleu1']
    comp = agg['composite_score']

    print(f"  {'Keyword F1':<30} {kf1['mean']:>8.3f} {kf1['std']:>8.3f} {kf1['min']:>8.3f} {kf1['max']:>8.3f}")
    print(f"  {'Cosine Similarity':<30} {css['mean']:>8.3f} {css['std']:>8.3f}")
    print(f"  {'BLEU-1':<30} {bl['mean']:>8.3f} {bl['std']:>8.3f}")
    print(f"  {'Composite Score':<30} {comp['mean']:>8.3f} {comp['std']:>8.3f}")
    print(f"  {'Retrieval Accuracy':<30} {agg['retrieval_accuracy']:>8.3f}")
    print(f"  {'Latency (ms)':<30} {agg['latency_ms']['mean']:>8.1f} {agg['latency_ms']['std']:>8.1f}")

    print(f"\n📋 PER-SAMPLE RESULTS")
    print(f"  {'ID':<6} {'KF1':>6} {'CSS':>6} {'BLEU':>6} {'Ret':>4} {'Comp':>6} {'ms':>6}")
    print(f"  {'-'*50}")
    for r in results:
        print(f"  {r.sample_id:<6} {r.keyword_f1:>6.3f} {r.cosine_similarity:>6.3f} "
              f"{r.bleu1_score:>6.3f} {'✓' if r.retrieval_hit else '✗':>4} "
              f"{r.composite_score:>6.3f} {r.latency_ms:>6.0f}")

    # Best and worst
    sorted_by_comp = sorted(results, key=lambda x: x.composite_score, reverse=True)
    print(f"\n🏆 Best Answer:  {sorted_by_comp[0].sample_id} (composite={sorted_by_comp[0].composite_score:.3f})")
    print(f"   Q: {sorted_by_comp[0].question[:60]}...")
    print(f"\n⚠️  Worst Answer: {sorted_by_comp[-1].sample_id} (composite={sorted_by_comp[-1].composite_score:.3f})")
    print(f"   Q: {sorted_by_comp[-1].question[:60]}...")
    print("\n" + "="*70)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from rag_pipeline import RAGPipeline

    rag = RAGPipeline(dataset_dir="../dataset", chunk_size=400, chunk_overlap=80, top_k=5)
    rag.build_index()

    evaluator = RAGEvaluator(rag, qa_path="../dataset/qa_pairs.json")
    results = evaluator.run_evaluation()
    agg = evaluator.aggregate_metrics(results)
    print_evaluation_report(results, agg)

    output = evaluator.save_results("../results/evaluation_results.json", results)

    print("\n\n=== QUALITATIVE ASSESSMENT INTERFACE ===")
    print_qualitative_assessment(results, n_sample=3)
