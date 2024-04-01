"""
Care-Plan Suggestion Agent — Node 4.

Generates structured next-step suggestions based on severity + DSM context.

IMPORTANT: Suggestions are NOT prescriptions or diagnoses.  They are structured
pointers to evidence-based pathways for a clinician to consider.  Every suggestion
must be prefaced with "Consider:" and include a citation or source.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import AgentState

SYSTEM_PROMPT = """You are a clinical decision support assistant helping a psychiatrist \
think through next steps for a patient. Based on the screening data, symptom signals, \
and retrieved clinical context provided below, generate a SHORT list of evidence-based \
next-step suggestions.

Rules you MUST follow:
1. Every suggestion MUST start with "Consider:".
2. Each suggestion must reference a clinical basis (guideline, DSM criterion, or citation).
3. Do NOT suggest specific medications, doses, or diagnoses — that is the clinician's role.
4. Do NOT generate more than 5 suggestions.
5. If suicidal ideation is positive, the FIRST suggestion must be about safety planning.
6. Keep language clinical and concise — no hedging filler.

Return suggestions as a JSON array of strings, e.g.:
["Consider: ...", "Consider: ..."]
"""


class CarePlanAgent:
    def __init__(self, llm=None):
        self._llm = llm

    @classmethod
    def from_ollama(cls, model: str = "llama3.2") -> CarePlanAgent:
        try:
            from langchain_community.llms import Ollama
            return cls(llm=Ollama(model=model))
        except Exception:
            return cls(llm=None)

    def __call__(self, state: AgentState) -> dict:
        if self._llm is None:
            return {"care_plan_suggestions": self._rule_based(state)}

        context = _build_context(state)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=context),
        ]
        try:
            response = self._llm.invoke(messages)
            content = response.content if hasattr(response, "content") else str(response)
            suggestions = _parse_suggestions(content)
        except Exception:
            suggestions = self._rule_based(state)

        return {"care_plan_suggestions": suggestions}

    def _rule_based(self, state: AgentState) -> list[str]:
        suggestions = []
        si_positive = state.get("suicidal_ideation_positive", False)
        band = state.get("phq9_band", "none")

        if si_positive:
            suggestions.append(
                "Consider: Immediate safety assessment using Columbia Suicide Severity "
                "Rating Scale (C-SSRS). Determine plan, intent, and means access."
            )

        if band in ("moderate", "moderately_severe", "severe"):
            suggestions.append(
                "Consider: Referral for psychiatric evaluation and evidence-based therapy "
                "(CBT or IPT) per NICE Depression Guidelines (CG90)."
            )

        if state.get("gad7_band", "minimal") in ("moderate", "severe"):
            suggestions.append(
                "Consider: GAD management pathway — CBT, relaxation techniques, and "
                "evaluation for comorbid depression per NICE GAD Guideline (CG113)."
            )

        if band in ("mild", "moderate"):
            suggestions.append(
                "Consider: Follow-up PHQ-9 reassessment in 2–4 weeks to monitor trajectory."
            )

        if not suggestions:
            suggestions.append(
                "Consider: Routine monitoring with PHQ-9 at next appointment. "
                "Psychoeducation on depression and anxiety management strategies."
            )

        return suggestions


def _build_context(state: AgentState) -> str:
    return (
        f"PHQ-9 total: {state.get('phq9_total', 'N/A')} ({state.get('phq9_band', 'N/A')})\n"
        f"GAD-7 total: {state.get('gad7_total', 'N/A')} ({state.get('gad7_band', 'N/A')})\n"
        f"Suicidal ideation positive: {state.get('suicidal_ideation_positive', False)}\n"
        f"Active symptoms: {', '.join(state.get('active_symptoms', []))}\n"
        f"DSM context: {state.get('dsm_context', '')[:800]}\n"
        f"Citations: {len(state.get('citations', []))} retrieved\n"
    )


def _parse_suggestions(content: str) -> list[str]:
    import json
    import re

    content = content.strip()
    # Try to extract a JSON array from the response.
    match = re.search(r'\[.*?\]', content, re.DOTALL)
    if match:
        try:
            items = json.loads(match.group())
            if isinstance(items, list):
                return [str(i) for i in items if str(i).startswith("Consider:")]
        except json.JSONDecodeError:
            pass

    # Fallback: extract lines starting with "Consider:"
    lines = [line.strip() for line in content.split("\n")]
    return [line for line in lines if line.startswith("Consider:")][:5]
