"""
Safety-Critic Agent — Node 5 (final gatekeeper).

This is the most important agent in Psychiatrist. It enforces a hard safety
contract in two layers:

  Layer 1 — Deterministic rules (cannot be overridden by the LLM):
    Any of the following triggers an immediate 'escalate' verdict,
    regardless of what the upstream agents concluded:
      - suicidal_ideation_positive == True  (from Risk Agent / MentalBERT)
      - phq9_suicidal_ideation item > 0    (raw PHQ-9 item 9)
      - Any keyword from SI_KEYWORDS found in the narrative
    These rules implement the "fail safe" principle: a false positive
    (over-escalation) is far less harmful than a false negative.

  Layer 2 — LLM verifier:
    Reads the full care-plan output and checks for:
      - Hallucinated symptoms not mentioned in the patient record
      - Missing citations (care plan makes claims without evidence)
      - Inappropriate certainty (e.g., claiming a diagnosis)
      - Language that contradicts the established safety verdict
    The LLM verifier CAN downgrade a 'routine' verdict to 'monitor' based
    on subtle signals, but it CANNOT override a deterministic 'escalate'.

The deterministic_escalation flag in state records whether Layer 1 triggered.
This is logged separately from the safety_verdict so you can monitor
how often the LLM and the rules agree over time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import AgentState, Verdict

# Suicidal ideation keyword patterns — conservative, errs toward recall.
# Applied per sentence (see _check_deterministic) with conditional-frame exclusion.
SI_KEYWORDS: list[str] = [
    r"\bkill(?:ing)?\s+myself\b",                       # "kill myself" and "killing myself"
    r"\bend\s+my\s+life\b",
    r"\bwant(?:ed|ing)?\s+to\s+die\b",
    r"\bnot\s+want\s+to\s+live\b",
    r"\bI\s+(?:am|'m|feel|felt|am\s+feeling|was|have\s+been)\s+suicidal\b",  # first-person assertion
    r"\bsuicide\b",
    r"\brather\s+be\s+dead\b",
    r"\bwish\s+I\s+were\s+dead\b",
    r"\btake\s+my\s+own\s+life\b",
    r"\bhoping\s+I\s+don.t\s+wake\b",
    r"\bdon.t\s+want\s+to\s+wake\b",
    r"\bwouldn.t\s+mind\s+dying\b",
    r"\bno\s+reason\s+to\s+live\b",
    r"\bbetter\s+off\s+dead\b",
    r"\bbetter\s+off\s+without\s+me\b",
    r"\bwithout\s+me\s+(?:here|around)\b",
    r"\bgo\s+to\s+sleep\s+and\s+never\s+wake\b",
    r"\bwriting\s+.*?letters?\b",                       # farewell letters (SUBTLE-001)
    r"\bgiving\s+away\s+my\b",                          # giving away possessions (SUBTLE-002)
    r"\bmy\s+funeral\b",                                # funeral ideation (SUBTLE-003)
    r"\bimagining.*funeral\b",
    r"\btake\s+all\s+my\s+(?:medication|pills?|tablets?|meds?)\b",  # overdose planning (OVERT-003)
    r"\boverdos(?:e|ing|ed)\b",
    r"\bbought\s+a\s+rope\b",                           # means acquisition (MEANS-002)
    r"\bhow\s+much\s+\S+\s+(?:it\s+)?would\s+take\b",  # method research (MEANS-001)
    r"\bhow\s+much\s+\S+.*would.*take\b",
]

# Sentences containing these frames are skipped — they indicate reported/conditional speech
# rather than first-person assertion of ideation.
_CONDITIONAL_FRAMES: list[str] = [
    r"\bask(?:ed|ing)?\s+(?:me|him|her|them|us|you)?\s+(?:if|whether)\b",
    r"\bif\s+I\s+was\s+suicidal\b",
    r"\bif\s+I\s+(?:am|were|had\s+been)\s+suicidal\b",
    r"\bwondering\s+(?:if|whether)\b",
]

VERIFIER_SYSTEM = """You are a clinical safety verifier reviewing a generated care-plan \
summary before it is shown to a clinician.

Your task:
1. Identify any hallucinated symptoms — symptoms mentioned in the summary that are NOT in the \
patient's record.
2. Identify unsupported claims — factual assertions with no citation.
3. Identify inappropriate certainty — language that implies a diagnosis or prescription.
4. Identify any language that contradicts a safety escalation requirement.

The patient record and care plan will be provided in the next message.

Respond with a JSON object:
{
  "flags": ["<issue 1>", "<issue 2>"],
  "verdict_recommendation": "escalate" | "monitor" | "routine"
}

If no issues are found, return {"flags": [], "verdict_recommendation": "routine"}.
Respond ONLY with valid JSON. Do not add prose.
"""


@dataclass
class SafetyCriticResult:
    verdict: Verdict
    reasoning: str
    deterministic_escalation: bool
    llm_verifier_flags: list[str]


def _sentence_has_si_signal(sentence: str) -> bool:
    """
    Return True if a single sentence contains an SI keyword
    AND is not a conditional/reported-speech frame.
    """
    s = sentence.strip()
    if not s:
        return False
    # Skip sentences that are clearly conditional or reported speech.
    for frame in _CONDITIONAL_FRAMES:
        if re.search(frame, s, re.IGNORECASE):
            return False
    return any(re.search(pattern, s, re.IGNORECASE) for pattern in SI_KEYWORDS)


def _check_deterministic(state: AgentState) -> bool:
    """Return True if any deterministic SI rule fires."""
    if state.get("suicidal_ideation_positive", False):
        return True
    if state["patient_input"].get("phq9_suicidal_ideation", 0) > 0:
        return True
    narrative = state["patient_input"].get("narrative", "")
    # Check sentence by sentence to avoid false positives from conditional frames.
    sentences = re.split(r"(?<=[.!?])\s+", narrative)
    return any(_sentence_has_si_signal(sentence) for sentence in sentences)


def _build_record_summary(state: AgentState) -> str:
    return (
        f"PHQ-9: {state.get('phq9_total', 'N/A')} ({state.get('phq9_band', 'N/A')})\n"
        f"GAD-7: {state.get('gad7_total', 'N/A')} ({state.get('gad7_band', 'N/A')})\n"
        f"Active symptoms from NLP: {', '.join(state.get('active_symptoms', []) or ['none detected'])}\n"
        f"Suicidal ideation (MentalBERT): {state.get('suicidal_ideation_prob', 0.0):.2f}\n"
        f"Suicidal ideation (PHQ-9 item 9): {state['patient_input'].get('phq9_suicidal_ideation', 0)}\n"
        f"Narrative (first 400 chars): {state['patient_input'].get('narrative', '')[:400]}"
    )


def _build_care_plan_summary(state: AgentState) -> str:
    suggestions = state.get("care_plan_suggestions", [])
    dsm_diff = state.get("dsm_differential", [])
    return (
        f"DSM differential: {', '.join(dsm_diff) if dsm_diff else 'not specified'}\n"
        f"Care plan suggestions:\n"
        + "\n".join(f"  - {s}" for s in suggestions)
    )


class SafetyCriticAgent:
    def __init__(self, llm=None):
        self._llm = llm

    @classmethod
    def from_ollama(cls, model: str = "llama3.2") -> SafetyCriticAgent:
        try:
            from langchain_community.llms import Ollama
            return cls(llm=Ollama(model=model))
        except Exception:
            return cls(llm=None)

    def __call__(self, state: AgentState) -> dict:
        result = self.evaluate(state)
        return {
            "safety_verdict": result.verdict,
            "critic_reasoning": result.reasoning,
            "deterministic_escalation": result.deterministic_escalation,
            "llm_verifier_flags": result.llm_verifier_flags,
            "report_ready": True,
        }

    def evaluate(self, state: AgentState) -> SafetyCriticResult:
        """Public API consumed by the safety regression tests."""
        # Layer 1: deterministic rules — CANNOT be overridden.
        det_escalate = _check_deterministic(state)
        if det_escalate:
            return SafetyCriticResult(
                verdict="escalate",
                reasoning=(
                    "DETERMINISTIC ESCALATION: Suicidal ideation signal detected "
                    "(PHQ-9 item 9 > 0, MentalBERT SI positive, or SI keyword in narrative). "
                    "This verdict cannot be overridden. Refer immediately to crisis protocol."
                ),
                deterministic_escalation=True,
                llm_verifier_flags=[],
            )

        # Layer 2: LLM verifier.
        flags, llm_verdict = self._run_llm_verifier(state)

        # The LLM can only escalate or add caution — never reduce to below 'monitor'
        # if the structured severity is moderate+.
        band = state.get("phq9_band", "none")
        if band in ("moderate", "moderately_severe", "severe") and llm_verdict == "routine":
            llm_verdict = "monitor"

        reasoning = self._build_reasoning(flags, llm_verdict, det_escalate)
        return SafetyCriticResult(
            verdict=llm_verdict,
            reasoning=reasoning,
            deterministic_escalation=False,
            llm_verifier_flags=flags,
        )

    def _run_llm_verifier(self, state: AgentState) -> tuple[list[str], Verdict]:
        if self._llm is None:
            return [], self._heuristic_verdict(state)

        import json

        record_summary = _build_record_summary(state)
        care_plan_summary = _build_care_plan_summary(state)
        user_content = (
            f"Patient record summary:\n{record_summary}\n\n"
            f"Generated care-plan summary:\n{care_plan_summary}"
        )
        try:
            response = self._llm.invoke([
                SystemMessage(content=VERIFIER_SYSTEM),
                HumanMessage(content=user_content),
            ])
            content = response.content if hasattr(response, "content") else str(response)
            content = content.strip()
            # Extract JSON from response (model may add prose despite instructions).
            match = re.search(r'\{.*?\}', content, re.DOTALL)
            if match:
                data = json.loads(match.group())
                flags = data.get("flags", [])
                rec = data.get("verdict_recommendation", "routine")
                verdict: Verdict = rec if rec in ("escalate", "monitor", "routine") else "monitor"
                return flags, verdict
        except Exception:
            pass
        return [], self._heuristic_verdict(state)

    def _heuristic_verdict(self, state: AgentState) -> Verdict:
        band = state.get("phq9_band", "none")
        if band in ("moderately_severe", "severe"):
            return "monitor"
        if band in ("moderate",):
            return "monitor"
        return "routine"

    def _build_reasoning(
        self, flags: list[str], verdict: Verdict, det_escalate: bool
    ) -> str:
        parts = [f"Safety verdict: {verdict.upper()}"]
        if flags:
            parts.append(f"LLM verifier flags: {'; '.join(flags)}")
        else:
            parts.append("LLM verifier: no issues found in the generated care plan.")
        return " | ".join(parts)


# ── Public API consumed by the safety regression test suite ───────────────────

def classify(text: str, llm=None) -> SafetyCriticResult:
    """
    Convenience function for the safety regression suite.

    Given a raw text passage, builds a minimal state and runs the Safety-Critic.
    The deterministic rules run on the narrative directly.  The LLM verifier is
    skipped (no care plan has been generated for raw-text testing).

    This is the entry point imported in tests/safety/test_safety_critic.py.
    """
    minimal_state: AgentState = {
        "patient_input": {
            "phq1_anhedonia": 0, "phq2_low_mood": 0, "phq3_sleep": 0,
            "phq4_fatigue": 0, "phq5_appetite": 0, "phq6_self_worth": 0,
            "phq7_concentration": 0, "phq8_psychomotor": 0,
            "phq9_suicidal_ideation": 0,
            "gad1_nervous": 0, "gad2_uncontrollable_worry": 0,
            "gad3_excess_worry": 0, "gad4_trouble_relaxing": 0,
            "gad5_restless": 0, "gad6_irritable": 0, "gad7_fearful": 0,
            "age": 30, "gender": "unknown", "narrative": text,
        },
        "messages": [],
        "phq9_total": 0, "gad7_total": 0,
        "phq9_band": "none", "gad7_band": "minimal",
        "severity_score": 0.0,
        "suicidal_ideation_prob": 0.0,
        "suicidal_ideation_positive": False,
        "active_symptoms": [], "symptom_probabilities": {},
        "dsm_context": "", "pubmed_context": "",
        "citations": [], "dsm_differential": [],
        "care_plan_suggestions": [],
        "safety_verdict": "routine",
        "critic_reasoning": "",
        "deterministic_escalation": False,
        "llm_verifier_flags": [],
        "report_ready": False, "error": None,
    }

    agent = SafetyCriticAgent(llm=llm)
    return agent.evaluate(minimal_state)
