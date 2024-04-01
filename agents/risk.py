"""
Risk-Stratification Agent — Node 2.

Fuses the structured PHQ-9 severity score (from Screening Agent) with
the MentalBERT NLP signals from the patient narrative.

Outputs a refined severity_score, suicidal_ideation_prob,
suicidal_ideation_positive, and active_symptoms.
"""

from __future__ import annotations

from agents.state import AgentState


class RiskAgent:
    def __init__(self, nlp_predictor=None):
        """
        Args:
            nlp_predictor: ClinicalNLPPredictor instance.
                           If None, the agent falls back to keyword heuristics only.
        """
        self._nlp = nlp_predictor

    @classmethod
    def from_checkpoints(cls, checkpoint_root: str = "models/clinical_nlp/checkpoints"):
        try:
            from models.clinical_nlp.predict import ClinicalNLPPredictor
            predictor = ClinicalNLPPredictor.from_checkpoints(checkpoint_root)
            return cls(nlp_predictor=predictor)
        except Exception:
            return cls(nlp_predictor=None)

    def __call__(self, state: AgentState) -> dict:
        narrative = state["patient_input"].get("narrative", "")
        structured_score = state.get("severity_score", 0.0)
        phq9_si_item = state["patient_input"].get("phq9_suicidal_ideation", 0)

        # Deterministic fallback: PHQ-9 item 9 is always checked regardless of NLP.
        item9_positive = phq9_si_item > 0

        if self._nlp and narrative:
            result = self._nlp.predict(narrative)
            si_prob = result.suicidal_ideation_prob
            si_positive = result.suicidal_ideation_positive or item9_positive
            active_symptoms = result.active_symptoms
            symptom_probs = result.symptoms
            # Blend structured + text severity (weighted average).
            text_severity = si_prob * 0.3 + min(len(active_symptoms) / 7.0, 1.0) * 0.2
            blended_score = 0.6 * structured_score + 0.4 * text_severity
        else:
            # Keyword fallback — used during dev before models are trained.
            si_keywords = [
                "kill myself", "end my life", "want to die", "suicidal",
                "not want to live", "rather be dead", "wish I were dead",
            ]
            text_lower = narrative.lower()
            kw_si = any(kw in text_lower for kw in si_keywords)
            si_positive = item9_positive or kw_si
            si_prob = 0.9 if si_positive else 0.05
            active_symptoms = []
            symptom_probs = {}
            blended_score = structured_score

        return {
            "suicidal_ideation_prob": si_prob,
            "suicidal_ideation_positive": si_positive,
            "active_symptoms": active_symptoms,
            "symptom_probabilities": symptom_probs,
            "severity_score": blended_score,
        }
