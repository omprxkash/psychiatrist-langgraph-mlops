"""
LCEL RAG chain for Psychiatrist.

Chain contract:
  - Input: { "query": str, "source_filter": list[str] | None }
  - Output: { "answer": str, "citations": list[dict], "context_chunks": list[str] }
  - Hard rule: if the retriever returns 0 chunks, the chain returns a refusal
    rather than hallucinating an answer.

The chain is consumed by the DSM/Literature Agent in agents/dsm_literature.py.
"""

from __future__ import annotations

from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from rag.retriever import HybridRetriever

SYSTEM_PROMPT = """You are a clinical information assistant supporting a psychiatrist.
Your role is to summarise the most relevant information from retrieved context for a \
clinical question. You MUST:
  1. Base your answer ONLY on the retrieved context provided below.
  2. Cite the source of every factual claim using the format [Source: <source>, PMID: <pmid>] \
or [Source: dsm, Disorder: <disorder>].
  3. If the context does not contain enough information to answer the question, say so \
explicitly — do NOT fabricate information.
  4. Keep your answer concise (3–6 sentences). Clinicians need brevity.

Retrieved context:
{context}
"""

USER_PROMPT = "{query}"

NO_CONTEXT_RESPONSE = (
    "I could not retrieve any relevant information from the indexed sources "
    "(DSM summaries and PubMed abstracts) for this query. "
    "Please consult a licensed psychiatrist and primary literature directly."
)


def build_rag_chain(retriever: HybridRetriever, llm: Any) -> Any:
    """
    Build a LangChain LCEL RAG chain.

    Args:
        retriever: HybridRetriever over DSM + PubMed indexes.
        llm: any LangChain-compatible chat LLM (Ollama, HuggingFacePipeline, etc.)

    Returns:
        A runnable chain. Call with:
            chain.invoke({"query": "...", "source_filter": None})
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
    )

    def _retrieve_and_format(inputs: dict) -> dict:
        query = inputs["query"]
        context, citations = retriever.retrieve_with_sources(query)
        if not context:
            return {
                "query": query,
                "context": "",
                "citations": [],
                "context_chunks": [],
                "_no_context": True,
            }
        return {
            "query": query,
            "context": context,
            "citations": citations,
            "context_chunks": [c["title"] or c["disorder"] or "" for c in citations],
            "_no_context": False,
        }

    def _guard_no_context(inputs: dict) -> dict:
        if inputs.get("_no_context"):
            return {**inputs, "answer": NO_CONTEXT_RESPONSE}
        return inputs

    def _run_llm(inputs: dict) -> dict:
        if inputs.get("_no_context"):
            return inputs
        messages = prompt.format_messages(context=inputs["context"], query=inputs["query"])
        response = llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)
        return {**inputs, "answer": answer}

    def _format_output(inputs: dict) -> dict:
        return {
            "answer": inputs.get("answer", NO_CONTEXT_RESPONSE),
            "citations": inputs.get("citations", []),
            "context_chunks": inputs.get("context_chunks", []),
        }

    chain = (
        RunnableLambda(_retrieve_and_format)
        | RunnableLambda(_guard_no_context)
        | RunnableLambda(_run_llm)
        | RunnableLambda(_format_output)
    )

    return chain
