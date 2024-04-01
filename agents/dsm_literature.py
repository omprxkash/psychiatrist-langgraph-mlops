"""
DSM/Literature Agent — Node 3.

Builds a query from the Risk Agent's outputs and retrieves relevant
DSM-5-TR summaries + PubMed abstracts via the hybrid RAG chain.
"""

from __future__ import annotations

from agents.state import AgentState


def _build_query(state: AgentState) -> str:
    parts = []
    phq9_band = state.get("phq9_band", "")
    gad7_band = state.get("gad7_band", "")
    symptoms = state.get("active_symptoms", [])
    si_positive = state.get("suicidal_ideation_positive", False)

    if phq9_band in ("moderate", "moderately_severe", "severe"):
        parts.append("Major Depressive Disorder")
    elif phq9_band == "mild":
        parts.append("mild depression dysthymia")

    if gad7_band in ("moderate", "severe"):
        parts.append("Generalised Anxiety Disorder")

    if si_positive:
        parts.append("suicidal ideation risk assessment escalation")

    parts.extend(symptoms[:3])
    return " ".join(parts) or "depression anxiety mental health"


class DSMLiteratureAgent:
    def __init__(self, rag_chain=None):
        self._chain = rag_chain

    @classmethod
    def from_indexes(
        cls,
        index_root: str = "rag/indexes",
        llm=None,
    ) -> DSMLiteratureAgent:
        from pathlib import Path

        from rag.chain import build_rag_chain
        from rag.retriever import HybridRetriever

        index_dirs = {
            "dsm": Path(index_root) / "dsm",
            "pubmed": Path(index_root) / "pubmed",
        }
        existing = {k: v for k, v in index_dirs.items() if (v / "faiss.index").exists()}
        if not existing:
            return cls(rag_chain=None)

        retriever = HybridRetriever.from_index_dirs(existing)

        if llm is None:
            try:
                from langchain_community.llms import Ollama
                llm = Ollama(model="llama3.2")
            except Exception:
                return cls(rag_chain=None)

        return cls(rag_chain=build_rag_chain(retriever, llm))

    def __call__(self, state: AgentState) -> dict:
        query = _build_query(state)

        if self._chain is None:
            return {
                "dsm_context": "RAG index not available — run `make rag-index` first.",
                "pubmed_context": "",
                "citations": [],
                "dsm_differential": [],
            }

        result = self._chain.invoke({"query": query})
        answer = result.get("answer", "")
        citations = result.get("citations", [])

        dsm_differential = [
            c["disorder"] for c in citations
            if c.get("source") == "dsm" and c.get("disorder")
        ]

        return {
            "dsm_context": answer,
            "pubmed_context": "",
            "citations": citations,
            "dsm_differential": dsm_differential,
        }
