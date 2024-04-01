"""
LangGraph shared state for the Psychiatrist agent pipeline.

All agents read from and write to this TypedDict.  LangGraph merges
updates from each node into the running state object.
"""

from __future__ import annotations

from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

Verdict = Literal["escalate", "monitor", "routine"]


class PatientInput(TypedDict):
    """Raw input from the intake form."""
    phq1_anhedonia: int
    phq2_low_mood: int
    phq3_sleep: int
    phq4_fatigue: int
    phq5_appetite: int
    phq6_self_worth: int
    phq7_concentration: int
    phq8_psychomotor: int
    phq9_suicidal_ideation: int
    gad1_nervous: int
    gad2_uncontrollable_worry: int
    gad3_excess_worry: int
    gad4_trouble_relaxing: int
    gad5_restless: int
    gad6_irritable: int
    gad7_fearful: int
    age: int
    gender: str
    narrative: str  # free-text patient note or session narrative


class AgentState(TypedDict):
    # Raw intake
    patient_input: PatientInput
    messages: Annotated[list[BaseMessage], add_messages]

    # Screening Agent outputs
    phq9_total: int
    gad7_total: int
    phq9_band: str
    gad7_band: str

    # Risk Agent outputs
    severity_score: float          # 0–1 normalised
    suicidal_ideation_prob: float  # from MentalBERT SI classifier
    suicidal_ideation_positive: bool
    active_symptoms: list[str]
    symptom_probabilities: dict[str, float]

    # DSM/Literature Agent outputs
    dsm_context: str
    pubmed_context: str
    citations: list[dict]
    dsm_differential: list[str]

    # Care-Plan Agent outputs
    care_plan_suggestions: list[str]

    # Safety-Critic outputs
    safety_verdict: Verdict
    critic_reasoning: str
    deterministic_escalation: bool  # True if rules triggered, independent of LLM
    llm_verifier_flags: list[str]

    # Final report
    report_ready: bool
    error: str | None
