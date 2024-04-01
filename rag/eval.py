"""
RAGAS evaluation scaffold for the RAG chain.

Evaluates: faithfulness, answer_relevance, context_precision.

Usage:
    python -m rag.eval --index-dir rag/indexes --n 50

Requires:
    pip install ragas datasets
    Ollama running locally with llama3.2 pulled.
"""

from __future__ import annotations

import argparse
from pathlib import Path

EVAL_QUESTIONS = [
    {
        "question": "What are the DSM-5 diagnostic criteria for Major Depressive Disorder?",
        "ground_truth": (
            "Five or more depressive symptoms present for at least 2 weeks, "
            "with at least one being depressed mood or loss of interest."
        ),
    },
    {
        "question": "What PHQ-9 score threshold indicates moderate depression?",
        "ground_truth": "A PHQ-9 total score of 10-14 indicates moderate depression.",
    },
    {
        "question": "What are the risk factors for suicide ideation escalation?",
        "ground_truth": (
            "Risk factors include specific plan, access to means, prior attempts, "
            "recent losses, hopelessness, and substance use comorbidity."
        ),
    },
    {
        "question": "How does Generalised Anxiety Disorder differ from Major Depressive Disorder?",
        "ground_truth": (
            "GAD is characterised by excessive worry about multiple domains lasting "
            "6+ months; MDD is characterised by depressed mood or anhedonia for 2+ weeks."
        ),
    },
    {
        "question": "What is the recommended first-line treatment for moderate-to-severe depression?",
        "ground_truth": (
            "Combination of antidepressant medication and psychotherapy, "
            "particularly CBT, is typically recommended."
        ),
    },
]


def run_eval(chain, index_dirs: dict[str, Path], n: int = 50):
    """Run RAGAS evaluation on a sample of questions."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
    except ImportError:
        print("ragas not installed. Run: pip install ragas datasets")
        return

    questions = (EVAL_QUESTIONS * ((n // len(EVAL_QUESTIONS)) + 1))[:n]

    records = []
    for q in questions:
        result = chain.invoke({"query": q["question"]})
        records.append(
            {
                "question": q["question"],
                "answer": result["answer"],
                "contexts": result["context_chunks"],
                "ground_truth": q["ground_truth"],
            }
        )

    dataset = Dataset.from_list(records)
    scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
    print("\nRAGAS scores:")
    for metric, val in scores.items():
        print(f"  {metric}: {val:.4f}")

    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", type=Path, default=Path("rag/indexes"))
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()

    from rag.chain import build_rag_chain
    from rag.retriever import HybridRetriever

    index_dirs = {
        "dsm": args.index_dir / "dsm",
        "pubmed": args.index_dir / "pubmed",
    }
    existing = {k: v for k, v in index_dirs.items() if (v / "faiss.index").exists()}
    if not existing:
        print(f"No indexes found in {args.index_dir}. Run `make rag-index` first.")
        return

    retriever = HybridRetriever.from_index_dirs(existing)

    try:
        from langchain_community.llms import Ollama
        llm = Ollama(model="llama3.2")
    except Exception as e:
        print(f"Could not connect to Ollama ({e}). Make sure Ollama is running.")
        return

    chain = build_rag_chain(retriever, llm)
    run_eval(chain, existing, n=args.n)


if __name__ == "__main__":
    main()
