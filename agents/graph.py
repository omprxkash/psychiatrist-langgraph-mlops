"""
LangGraph state machine definition for Psychiatrist.

Pipeline:
    START → screening → risk → dsm_literature → care_plan → safety_critic → END

The safety_critic node has no conditional edges — it always runs last and its
verdict is embedded in the final state.  The FastAPI layer decides what to do
with an 'escalate' verdict (surface a red banner, notify clinician, etc.).

This separation is intentional: the graph is deterministic from the caller's
perspective, and escalation logic lives in a single, testable node rather than
scattered across routing logic.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.care_plan import CarePlanAgent
from agents.dsm_literature import DSMLiteratureAgent
from agents.risk import RiskAgent
from agents.safety_critic import SafetyCriticAgent
from agents.screening import ScreeningAgent
from agents.state import AgentState


def build_graph(
    severity_clf=None,
    nlp_predictor=None,
    rag_chain=None,
    llm=None,
) -> StateGraph:
    """
    Build and compile the Psychiatrist LangGraph workflow.

    All dependencies are injected so each component can be swapped for a mock
    during testing without touching the graph topology.

    Args:
        severity_clf:  sklearn-compatible severity classifier (None → rules-only fallback).
        nlp_predictor: ClinicalNLPPredictor (None → keyword fallback).
        rag_chain:     LangChain LCEL chain (None → returns placeholder context).
        llm:           LangChain-compatible LLM for care-plan + safety-critic verifier.
    """
    graph = StateGraph(AgentState)

    screening = ScreeningAgent(severity_clf=severity_clf)
    risk = RiskAgent(nlp_predictor=nlp_predictor)
    dsm_lit = DSMLiteratureAgent(rag_chain=rag_chain)
    care_plan = CarePlanAgent(llm=llm)
    safety_critic = SafetyCriticAgent(llm=llm)

    graph.add_node("screening", screening)
    graph.add_node("risk", risk)
    graph.add_node("dsm_literature", dsm_lit)
    graph.add_node("care_plan", care_plan)
    graph.add_node("safety_critic", safety_critic)

    graph.add_edge(START, "screening")
    graph.add_edge("screening", "risk")
    graph.add_edge("risk", "dsm_literature")
    graph.add_edge("dsm_literature", "care_plan")
    graph.add_edge("care_plan", "safety_critic")
    graph.add_edge("safety_critic", END)

    return graph.compile()


def build_graph_from_env(
    index_root: str = "rag/indexes",
    checkpoint_root: str = "models/clinical_nlp/checkpoints",
    mlflow_model: str = "psychiatrist-severity-classifier",
    ollama_model: str = "llama3.2",
):
    """
    Convenience builder that loads all dependencies from their default paths.
    Used by the FastAPI serving layer.
    """
    from langchain_community.llms import Ollama

    llm = None
    try:
        llm = Ollama(model=ollama_model)
    except Exception as e:
        print(f"Ollama not available ({e}). LLM-dependent agents will use heuristics.")

    nlp_predictor = None
    try:
        from models.clinical_nlp.predict import ClinicalNLPPredictor
        nlp_predictor = ClinicalNLPPredictor.from_checkpoints(checkpoint_root)
    except Exception as e:
        print(f"NLP predictor not available ({e}). Using keyword fallback.")

    rag_chain = None
    try:
        from pathlib import Path

        from rag.chain import build_rag_chain
        from rag.retriever import HybridRetriever

        index_dirs = {
            "dsm": Path(index_root) / "dsm",
            "pubmed": Path(index_root) / "pubmed",
        }
        existing = {k: v for k, v in index_dirs.items() if (v / "faiss.index").exists()}
        if existing and llm:
            retriever = HybridRetriever.from_index_dirs(existing)
            rag_chain = build_rag_chain(retriever, llm)
    except Exception as e:
        print(f"RAG chain not available ({e}). DSM context will be placeholder.")

    severity_clf = None
    try:
        import mlflow
        severity_clf = mlflow.sklearn.load_model(f"models:/{mlflow_model}/Production")
    except Exception:
        pass

    return build_graph(
        severity_clf=severity_clf,
        nlp_predictor=nlp_predictor,
        rag_chain=rag_chain,
        llm=llm,
    )
