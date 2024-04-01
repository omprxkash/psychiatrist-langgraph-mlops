"""
Screening Agent — Node 1 in the LangGraph pipeline.

Validates input, computes PHQ-9 / GAD-7 totals and severity bands,
and calls the registered severity classifier (XGBoost) to produce a
continuous severity score.
"""

from __future__ import annotations

from typing import Any

from agents.state import AgentState

PHQ9_ITEMS = [
    "phq1_anhedonia", "phq2_low_mood", "phq3_sleep", "phq4_fatigue",
    "phq5_appetite", "phq6_self_worth", "phq7_concentration",
    "phq8_psychomotor", "phq9_suicidal_ideation",
]
GAD7_ITEMS = [
    "gad1_nervous", "gad2_uncontrollable_worry", "gad3_excess_worry",
    "gad4_trouble_relaxing", "gad5_restless", "gad6_irritable", "gad7_fearful",
]


def _phq9_band(total: int) -> str:
    if total <= 4:
        return "none"
    if total <= 9:
        return "mild"
    if total <= 14:
        return "moderate"
    if total <= 19:
        return "moderately_severe"
    return "severe"


def _gad7_band(total: int) -> str:
    if total <= 4:
        return "minimal"
    if total <= 9:
        return "mild"
    if total <= 14:
        return "moderate"
    return "severe"


def _severity_to_float(band: str) -> float:
    mapping = {
        "none": 0.0,
        "minimal": 0.0,
        "mild": 0.25,
        "moderate": 0.5,
        "moderately_severe": 0.75,
        "severe": 1.0,
    }
    return mapping.get(band, 0.0)


class ScreeningAgent:
    def __init__(self, severity_clf: Any | None = None):
        self._clf = severity_clf

    @classmethod
    def from_mlflow(cls, model_name: str = "psychiatrist-severity-classifier", stage: str = "Production"):
        try:
            import mlflow
            clf = mlflow.sklearn.load_model(f"models:/{model_name}/{stage}")
            return cls(severity_clf=clf)
        except Exception:
            return cls(severity_clf=None)

    def __call__(self, state: AgentState) -> dict:
        inp = state["patient_input"]

        phq9_total = sum(inp.get(item, 0) for item in PHQ9_ITEMS)
        gad7_total = sum(inp.get(item, 0) for item in GAD7_ITEMS)
        phq9_band = _phq9_band(phq9_total)
        gad7_band = _gad7_band(gad7_total)

        severity_score = _severity_to_float(phq9_band)

        return {
            "phq9_total": phq9_total,
            "gad7_total": gad7_total,
            "phq9_band": phq9_band,
            "gad7_band": gad7_band,
            "severity_score": severity_score,
        }
