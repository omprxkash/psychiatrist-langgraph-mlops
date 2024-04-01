"""
Reusable Streamlit UI components for Psychiatrist.

Keeps app.py readable by pulling display logic into focused helpers.
"""

from __future__ import annotations

# isort: off
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # backend must be set before pyplot import
import numpy as np
# isort: on


# ── Colour palette — Limitless design tokens (oklch) ──────────────────────────
# Neutral spine: n50 (lightest) → n900 (darkest)
# Teal accent for routine/safe; red-500 reserved for escalate only.

BAND_COLOURS = {
    "none":               ("oklch(97% 0 0)",         "oklch(37.1% 0 0)"),   # neutral-100 / neutral-700
    "minimal":            ("oklch(97% 0 0)",         "oklch(37.1% 0 0)"),
    "mild":               ("oklch(97.5% .025 90)",   "oklch(38% .08 65)"),  # very light amber / amber-dark
    "moderate":           ("oklch(96% .04 55)",      "oklch(35% .1 45)"),   # light orange tint
    "moderately_severe":  ("oklch(96% .04 20)",      "oklch(35% .12 20)"),  # light rose tint
    "severe":             ("oklch(63.7% .237 25.331)", "#ffffff"),           # red-500 / white
}

GAD7_BAND_COLOURS = {
    "minimal":  ("oklch(97% 0 0)",           "oklch(37.1% 0 0)"),
    "mild":     ("oklch(97.5% .025 90)",     "oklch(38% .08 65)"),
    "moderate": ("oklch(96% .04 55)",        "oklch(35% .1 45)"),
    "severe":   ("oklch(63.7% .237 25.331)", "#ffffff"),
}

VERDICT_COLOURS = {
    "routine":  ("oklch(96% .05 182)", "oklch(38% .12 182)", "✓  Routine"),          # light teal / dark teal
    "monitor":  ("oklch(97.5% .025 90)", "oklch(38% .08 65)", "⚠  Monitor closely"), # light amber / dark amber
    "escalate": ("oklch(63.7% .237 25.331)", "#ffffff", "🚨  Escalate — urgent review required"),  # red-500 / white
}


def severity_card_html(band: str, phq9_total: int, gad7_total: int) -> str:
    bg, fg = BAND_COLOURS.get(band, ("#F0F0F0", "#333"))
    label = band.replace("_", " ").title()
    return f"""
    <div style="background:{bg}; color:{fg}; border-radius:0.75rem; padding:20px 24px;
                font-family:Inter,ui-sans-serif,sans-serif; letter-spacing:-0.02em;">
        <div style="font-size:0.72rem; font-weight:600; letter-spacing:0.06em;
                    text-transform:uppercase; opacity:0.65; margin-bottom:6px;">
            Depression severity
        </div>
        <div style="font-size:1.875rem; font-weight:700; letter-spacing:-0.04em;
                    line-height:1.2; margin-bottom:4px;">{label}</div>
        <div style="font-size:0.875rem; opacity:0.8;">
            PHQ-9&nbsp;{phq9_total} &nbsp;&middot;&nbsp; GAD-7&nbsp;{gad7_total}
        </div>
    </div>
    """


def anxiety_card_html(band: str, gad7_total: int) -> str:
    bg, fg = GAD7_BAND_COLOURS.get(band, ("#F0F0F0", "#333"))
    label = band.replace("_", " ").title()
    return f"""
    <div style="background:{bg}; color:{fg}; border-radius:0.75rem; padding:20px 24px;
                font-family:Inter,ui-sans-serif,sans-serif; letter-spacing:-0.02em;">
        <div style="font-size:0.72rem; font-weight:600; letter-spacing:0.06em;
                    text-transform:uppercase; opacity:0.65; margin-bottom:6px;">
            Anxiety severity
        </div>
        <div style="font-size:1.875rem; font-weight:700; letter-spacing:-0.04em;
                    line-height:1.2; margin-bottom:4px;">{label}</div>
        <div style="font-size:0.875rem; opacity:0.8;">
            GAD-7&nbsp;{gad7_total}
        </div>
    </div>
    """


def safety_verdict_html(verdict: str, escalation: bool, reasoning: str) -> str:
    bg, fg, label = VERDICT_COLOURS.get(verdict, ("#F0F0F0", "#333", verdict))
    border = "2px solid #721C24" if verdict == "escalate" else "none"
    flag_note = (
        "<div style='margin-top:8px; font-size:0.82rem; font-weight:600;'>"
        "Deterministic rule triggered (SI or imminent risk keywords detected).</div>"
        if escalation else ""
    )
    reasoning_html = (
        f"<div style='margin-top:8px; font-size:0.82rem; opacity:0.85;'>{reasoning}</div>"
        if reasoning else ""
    )
    return f"""
    <div style="background:{bg}; color:{fg}; border:{border}; border-radius:0.75rem;
                padding:16px 20px; font-family:Inter,ui-sans-serif,sans-serif; letter-spacing:-0.02em;">
        <div style="font-size:1rem; font-weight:700; letter-spacing:-0.02em;">{label}</div>
        {flag_note}
        {reasoning_html}
    </div>
    """


def feature_bar_chart(
    feature_names: list[str],
    values: list[float],
    title: str = "Feature contributions",
) -> plt.Figure:
    """
    Horizontal bar chart of feature contributions.  Positive values (higher
    score) are shown in the alert colour; zero values are muted grey.
    """
    n = len(feature_names)
    fig, ax = plt.subplots(figsize=(6, max(2.5, n * 0.45)))

    # Teal for contributions, neutral-300 for zero — reserve red for escalate state only
    colours = ["#2a9d8f" if v > 0 else "#d4d4d4" for v in values]
    y_pos = np.arange(n)
    ax.barh(y_pos, values, color=colours, height=0.55)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names, fontsize=9, fontfamily="sans-serif")
    ax.invert_yaxis()
    ax.set_xlabel("Contribution (normalised)", fontsize=9)
    ax.set_title(title, fontsize=10, pad=10, fontweight="600")
    ax.axvline(0, color="#a3a3a3", linewidth=0.7, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#e5e5e5")
    ax.tick_params(colors="#525252")
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafafa")

    fig.tight_layout()
    return fig


def phq9_item_contribution(patient_input: dict) -> tuple[list[str], list[float]]:
    """
    Heuristic feature contributions for display when no trained model is
    available.  Normalises raw item scores to [0, 1] and weights by clinical
    priority (SI > somatic > cognitive > motor).
    """
    items = [
        ("Anhedonia (Q1)",             patient_input.get("phq1_anhedonia", 0),        1.0),
        ("Low mood (Q2)",              patient_input.get("phq2_low_mood", 0),          1.0),
        ("Sleep disruption (Q3)",      patient_input.get("phq3_sleep", 0),             0.85),
        ("Fatigue (Q4)",               patient_input.get("phq4_fatigue", 0),           0.85),
        ("Appetite change (Q5)",       patient_input.get("phq5_appetite", 0),          0.8),
        ("Self-worth / guilt (Q6)",    patient_input.get("phq6_self_worth", 0),        0.9),
        ("Concentration (Q7)",         patient_input.get("phq7_concentration", 0),     0.8),
        ("Psychomotor change (Q8)",    patient_input.get("phq8_psychomotor", 0),       0.75),
        ("Suicidal ideation (Q9)",     patient_input.get("phq9_suicidal_ideation", 0), 1.5),
    ]

    names  = [name for name, _, _ in items]
    scores = [round((score / 3) * weight, 3) for _, score, weight in items]

    # Only top 6 by magnitude, descending
    paired = sorted(zip(scores, names, strict=False), reverse=True)[:6]
    top_scores = [s for s, _ in paired]
    top_names  = [n for _, n in paired]

    return top_names, top_scores


def care_plan_html(suggestions: list[str]) -> str:
    if not suggestions:
        return "<p style='color:#888;'>No care plan generated.</p>"
    items_html = "".join(
        f"<li style='margin-bottom:6px; line-height:1.5;'>{s}</li>"
        for s in suggestions
    )
    return f"""
    <ul style="list-style-type:disc; padding-left:1.4em;
               font-family:Inter,ui-sans-serif,sans-serif; font-size:0.9rem;
               line-height:1.65; color:oklch(37.1% 0 0); letter-spacing:-0.01em;">
      {items_html}
    </ul>
    """


def dsm_signals_html(signals: list[str]) -> str:
    if not signals:
        return "<p style='color:#888;'>No DSM-5 signals detected from narrative.</p>"
    tags = "".join(
        f"<span style='background:oklch(97% 0 0); color:oklch(37.1% 0 0);"
        f"border:1px solid oklch(87% 0 0); border-radius:0.375rem;"
        f"padding:3px 10px; margin:3px; display:inline-block; font-size:0.8rem;"
        f"font-family:Inter,sans-serif; letter-spacing:-0.01em;'>{s}</span>"
        for s in signals
    )
    return f"<div style='line-height:2.2;'>{tags}</div>"
