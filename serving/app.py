"""
Psychiatrist — clinical decision-support interface.

I built this as the front end of my mental-health triage project.
It takes a PHQ-9 + GAD-7 screening, a short narrative, and passes
everything through the LangGraph agent pipeline to produce a severity
assessment, safety verdict, and care-plan suggestions.

Run with:  streamlit run serving/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make sure the project root is on the path when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from serving.components import (
    anxiety_card_html,
    care_plan_html,
    dsm_signals_html,
    feature_bar_chart,
    phq9_item_contribution,
    safety_verdict_html,
    severity_card_html,
)

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Psychiatrist",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — muted clinical palette ───────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --c-white:   #ffffff;
        --c-black:   #000000;
        --c-n50:     oklch(98.5% 0 0);
        --c-n100:    oklch(97%   0 0);
        --c-n200:    oklch(92.2% 0 0);
        --c-n300:    oklch(87%   0 0);
        --c-n400:    oklch(70.8% 0 0);
        --c-n500:    oklch(55.6% 0 0);
        --c-n700:    oklch(37.1% 0 0);
        --c-n900:    oklch(20.5% 0 0);
        --c-teal4:   oklch(77.7% .152 181.912);
        --c-teal5:   oklch(70.4% .14  182.503);
        --c-teal6:   oklch(60%   .13  182.5);
        --c-red5:    oklch(63.7% .237 25.331);
        --radius-md: 0.375rem;
        --radius-xl: 0.75rem;
        --radius-2xl: 1rem;
        --transition: 0.15s cubic-bezier(.4,0,.2,1);
    }

    /* ── Global font ───────────────────────────────────────────────────────── */
    html, body, [class*="css"], .stApp, .stMarkdown, .stForm,
    button, input, select, textarea, label {
        font-family: Inter, ui-sans-serif, system-ui, sans-serif !important;
        letter-spacing: -0.02em;
        -webkit-font-smoothing: antialiased;
    }

    /* ── Page & layout ─────────────────────────────────────────────────────── */
    .stApp { background-color: var(--c-white); }
    section[data-testid="stSidebar"] { background-color: var(--c-n50); }
    .block-container { padding-top: 2rem; max-width: 72rem; }

    /* ── Typography ────────────────────────────────────────────────────────── */
    h1, h2, h3, h4, h5 {
        color: var(--c-n900);
        font-weight: 600;
        letter-spacing: -0.03em;
        line-height: 1.25;
    }
    p, li { color: var(--c-n700); line-height: 1.6; }

    /* ── Tabs ──────────────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        border-bottom: 1px solid var(--c-n200);
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: var(--radius-xl) var(--radius-xl) 0 0;
        padding: 10px 20px;
        font-weight: 500;
        font-size: 0.875rem;
        color: var(--c-n500);
        border: 1px solid transparent;
        border-bottom: none;
        transition: color var(--transition);
    }
    .stTabs [aria-selected="true"] {
        background: var(--c-white) !important;
        color: var(--c-n900) !important;
        border-color: var(--c-n200) !important;
        font-weight: 600 !important;
    }

    /* ── Submit button ─────────────────────────────────────────────────────── */
    div.stButton > button {
        background-color: var(--c-teal5);
        color: var(--c-white);
        border-radius: var(--radius-xl);
        padding: 11px 28px;
        font-weight: 600;
        font-size: 0.9rem;
        border: none;
        width: 100%;
        letter-spacing: -0.01em;
        transition: background-color var(--transition);
    }
    div.stButton > button:hover { background-color: var(--c-teal6); }

    /* ── Form inputs ───────────────────────────────────────────────────────── */
    .stSlider label      { font-size: 0.875rem; color: var(--c-n700); }
    .stTextArea textarea { border-radius: var(--radius-md); border-color: var(--c-n200); }
    .stNumberInput input { border-radius: var(--radius-md); }

    /* ── Metric tiles ──────────────────────────────────────────────────────── */
    [data-testid="metric-container"] {
        background: var(--c-n50);
        border: 1px solid var(--c-n200);
        border-radius: var(--radius-xl);
        padding: 12px 16px;
    }

    /* ── Disclaimer banner — sticky so it's visible on every tab ───────────── */
    .disclaimer-box {
        background: var(--c-n50);
        border-left: 3px solid var(--c-n300);
        border-radius: 0 var(--radius-xl) var(--radius-xl) 0;
        padding: 12px 16px;
        font-size: 0.82rem;
        color: var(--c-n700);
        margin-bottom: 16px;
        line-height: 1.5;
        position: sticky;
        top: 0;
        z-index: 100;
    }

    /* ── Semantic spacing ──────────────────────────────────────────────────── */
    .stack-gap { height: 12px; }

    /* ── Dividers ──────────────────────────────────────────────────────────── */
    hr { border-color: var(--c-n200); }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
<div style="padding: 8px 0 20px; border-bottom: 1px solid oklch(92.2% 0 0); margin-bottom: 20px;">
  <div style="font-size: 1.1rem; font-weight: 700; letter-spacing: -0.03em; color: oklch(20.5% 0 0);">
    Psychiatrist
  </div>
  <div style="font-size: 0.78rem; color: oklch(55.6% 0 0); margin-top: 3px; letter-spacing: -0.01em;">
    Mental health triage · portfolio project
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
A portfolio project on agentic clinical decision support. Combines a PHQ-9 / GAD-7
screening model, a LangGraph multi-agent pipeline, and a deterministic Safety-Critic.
""")

    st.markdown("""
<div style="margin-top: 20px; padding-top: 16px; border-top: 1px solid oklch(92.2% 0 0);">
  <div style="font-size: 0.75rem; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase;
              color: oklch(55.6% 0 0); margin-bottom: 10px;">
    Crisis lines — India
  </div>
  <div style="font-size: 0.8rem; color: oklch(37.1% 0 0); line-height: 1.8;">
    iCall &nbsp;·&nbsp; +91 9152987821<br>
    KIRAN &nbsp;·&nbsp; 1800-599-0019<br>
    Vandrevala &nbsp;·&nbsp; 1860-2662-345
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div style="margin-top: 20px; padding-top: 16px; border-top: 1px solid oklch(92.2% 0 0);
            font-size: 0.72rem; color: oklch(55.6% 0 0); line-height: 1.6;">
  XGBoost · MentalBERT (int8) · LangGraph · Ollama
</div>
""", unsafe_allow_html=True)

# ── PHQ-9 and GAD-7 question text ──────────────────────────────────────────────

PHQ9_QUESTIONS = [
    ("phq1_anhedonia",         "Little interest or pleasure in doing things"),
    ("phq2_low_mood",          "Feeling down, depressed, or hopeless"),
    ("phq3_sleep",             "Trouble falling or staying asleep, or sleeping too much"),
    ("phq4_fatigue",           "Feeling tired or having little energy"),
    ("phq5_appetite",          "Poor appetite or overeating"),
    ("phq6_self_worth",        "Feeling bad about yourself — or that you are a failure or have let yourself or your family down"),
    ("phq7_concentration",     "Trouble concentrating on things, such as reading or watching television"),
    ("phq8_psychomotor",       "Moving or speaking so slowly others noticed? Or the opposite — being so fidgety or restless that you moved around much more than usual"),
    ("phq9_suicidal_ideation", "Thoughts that you would be better off dead, or thoughts of hurting yourself"),
]

GAD7_QUESTIONS = [
    ("gad1_nervous",               "Feeling nervous, anxious, or on edge"),
    ("gad2_uncontrollable_worry",  "Not being able to stop or control worrying"),
    ("gad3_excess_worry",          "Worrying too much about different things"),
    ("gad4_trouble_relaxing",      "Trouble relaxing"),
    ("gad5_restless",              "Being so restless that it is hard to sit still"),
    ("gad6_irritable",             "Becoming easily annoyed or irritable"),
    ("gad7_fearful",               "Feeling afraid as if something awful might happen"),
]

RESPONSE_LABELS = {0: "Not at all", 1: "Several days", 2: "More than half the days", 3: "Nearly every day"}
_SLIDER_HELP = " · ".join(f"{k} = {v}" for k, v in RESPONSE_LABELS.items())


# ── Graph loader (cached across reruns) ───────────────────────────────────────

@st.cache_resource(show_spinner="Loading clinical pipeline…")
def load_graph():
    try:
        from agents.graph import build_graph_from_env
        return build_graph_from_env(), False
    except Exception as e:
        st.warning(f"Full pipeline unavailable ({e}). Running in heuristic fallback mode.")
        from agents.graph import build_graph
        return build_graph(), True


# ── Session state defaults ────────────────────────────────────────────────────

def _init_state():
    defaults = {k: 0 for k, _ in PHQ9_QUESTIONS + GAD7_QUESTIONS}
    defaults.update({
        "age": 30,
        "gender": "prefer_not_to_say",
        "prior_diagnosis": False,
        "narrative": "",
        "result": None,
        "submitted": False,
    })
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── Main UI ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="padding-bottom: 20px; border-bottom: 1px solid oklch(92.2% 0 0); margin-bottom: 24px;">
  <h1 style="font-size: 1.75rem; font-weight: 700; letter-spacing: -0.04em;
             color: oklch(20.5% 0 0); margin: 0 0 4px 0; line-height: 1.2;">
    Psychiatrist
  </h1>
  <p style="font-size: 0.9rem; color: oklch(55.6% 0 0); margin: 0; letter-spacing: -0.01em;">
    Mental health triage &amp; clinical decision support &mdash; portfolio project
  </p>
</div>
""", unsafe_allow_html=True)

tab_intake, tab_results, tab_literature = st.tabs(["📋 Patient intake", "📊 Clinical results", "📚 Literature"])

# ── Tab 1: Intake ──────────────────────────────────────────────────────────────

with tab_intake:
    with st.form("intake_form", clear_on_submit=False):
        st.markdown(
            "<h4 style='color:#1a1a1a; font-weight:600; letter-spacing:-0.03em;'>"
            "Over the last <strong>2 weeks</strong>, how often have you been bothered by the following?</h4>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size:0.85rem; color:#505050; margin:-8px 0 12px;'>"
            "0 = Not at all &nbsp;·&nbsp; 1 = Several days &nbsp;·&nbsp; "
            "2 = More than half the days &nbsp;·&nbsp; 3 = Nearly every day</p>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        st.markdown("<h5 style='color:#1a1a1a; font-weight:600;'>PHQ-9 — Depression screening</h5>", unsafe_allow_html=True)
        for key, question in PHQ9_QUESTIONS:
            # Q9 gets a visible warning label
            label = question if key != "phq9_suicidal_ideation" else f"⚠ {question}"
            st.session_state[key] = st.slider(
                label=label,
                min_value=0, max_value=3,
                value=st.session_state[key],
                format="%d",
                key=f"slider_{key}",
                help=_SLIDER_HELP,
            )

        st.markdown("---")
        st.markdown("<h5 style='color:#1a1a1a; font-weight:600;'>GAD-7 — Anxiety screening</h5>", unsafe_allow_html=True)
        for key, question in GAD7_QUESTIONS:
            st.session_state[key] = st.slider(
                label=question,
                min_value=0, max_value=3,
                value=st.session_state[key],
                format="%d",
                key=f"slider_{key}",
                help=_SLIDER_HELP,
            )

        st.markdown("---")
        st.markdown("<h5 style='color:#1a1a1a; font-weight:600;'>Demographics</h5>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state["age"] = st.number_input(
                "Age", min_value=18, max_value=100,
                value=st.session_state["age"], key="num_age",
            )
        with col2:
            st.session_state["gender"] = st.selectbox(
                "Gender",
                options=["female", "male", "non-binary", "prefer_not_to_say"],
                index=["female", "male", "non-binary", "prefer_not_to_say"].index(
                    st.session_state["gender"]
                ),
                key="sel_gender",
            )
        with col3:
            st.session_state["prior_diagnosis"] = st.checkbox(
                "Prior psychiatric diagnosis",
                value=st.session_state["prior_diagnosis"],
                key="chk_prior",
            )

        st.markdown("---")
        st.markdown("<h5 style='color:#1a1a1a; font-weight:600;'>Clinical narrative</h5>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.83rem; color:#505050; margin:-6px 0 8px;'>Brief description of presenting complaint, current functioning, or session notes.</p>", unsafe_allow_html=True)
        st.session_state["narrative"] = st.text_area(
            label="Narrative",
            value=st.session_state["narrative"],
            height=120,
            placeholder="e.g. Patient reports persistent low mood for three weeks with early-morning awakening and reduced appetite. Denies suicidal ideation.",
            key="ta_narrative",
            label_visibility="collapsed",
        )

        submitted = st.form_submit_button("Run clinical assessment →", use_container_width=True)

    if submitted:
        if not st.session_state["narrative"].strip():
            st.error("Please enter a brief clinical narrative before submitting.")
        else:
            # Build state dict and invoke graph
            patient_input = {
                **{k: st.session_state[k] for k, _ in PHQ9_QUESTIONS + GAD7_QUESTIONS},
                "age":    st.session_state["age"],
                "gender": st.session_state["gender"],
                "narrative": st.session_state["narrative"].strip(),
            }

            initial_state = {
                "patient_input": patient_input,
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
                "report_ready": False,
                "error": None,
            }

            with st.status("Running clinical assessment…", expanded=True) as _status:
                _status.write("🔍 Screening (PHQ-9 / GAD-7 scoring)")
                _status.write("⚠️  Risk assessment (SI classifier)")
                _status.write("📚 Literature retrieval (DSM-5 + PubMed RAG)")
                _status.write("💊 Care plan generation")
                _status.write("🛡️  Safety check (deterministic + LLM verifier)")
                try:
                    graph, _ = load_graph()
                    result = graph.invoke(initial_state)
                    st.session_state["result"] = result
                    st.session_state["submitted"] = True
                    _status.update(label="Assessment complete", state="complete", expanded=False)

                    # Write to audit log
                    try:
                        from monitoring.audit_log import log_prediction
                        log_prediction(patient_input, result)
                    except Exception:
                        pass  # audit log failure never blocks the UI

                    st.success("Assessment complete — see the Clinical results tab.")
                except Exception as e:
                    _status.update(label="Pipeline error — running heuristic fallback", state="error", expanded=False)
                    st.warning(f"Full pipeline unavailable ({type(e).__name__}). Retrying with heuristic fallback…")
                    try:
                        from agents.graph import build_graph
                        result = build_graph().invoke(initial_state)
                        st.session_state["result"] = result
                        st.session_state["submitted"] = True
                        st.info("Showing heuristic fallback results (no LLM / trained model).")
                    except Exception as e2:
                        st.error(f"Fallback also failed: {e2}. Check the terminal for details.")


# ── Tab 2: Results ─────────────────────────────────────────────────────────────

with tab_results:
    result = st.session_state.get("result")

    if result is None:
        st.info("Complete the intake form and submit to see results here.")
    else:
        # Full-width escalate banner — rendered before the two-column grid so it's never below fold
        if result.get("safety_verdict") == "escalate":
            st.markdown(
                safety_verdict_html(
                    "escalate",
                    result.get("deterministic_escalation", False),
                    result.get("critic_reasoning", ""),
                ),
                unsafe_allow_html=True,
            )
            st.markdown("""
<div style="background:oklch(96% .04 20); border-left:3px solid oklch(63.7% .237 25.331);
            border-radius:0 0.75rem 0.75rem 0; padding:12px 16px; font-size:0.85rem;
            color:oklch(35% .12 20); margin-bottom:16px; font-family:Inter,sans-serif;">
  <strong>Crisis lines India:</strong>
  iCall +91&nbsp;9152987821 &nbsp;·&nbsp; KIRAN 1800-599-0019 &nbsp;·&nbsp; Vandrevala 1860-2662-345
  <br><strong>International:</strong> Crisis Text Line — text HOME to 741741 (US/CA/UK/IE)
</div>
""", unsafe_allow_html=True)
            st.markdown("<div class='stack-gap'></div>", unsafe_allow_html=True)

        col_left, col_right = st.columns([1, 1], gap="large")

        with col_left:
            st.markdown("#### Severity assessment")
            sev_col, anx_col = st.columns(2)
            with sev_col:
                st.markdown(
                    severity_card_html(
                        result.get("phq9_band", "none"),
                        result.get("phq9_total", 0),
                        result.get("gad7_total", 0),
                    ),
                    unsafe_allow_html=True,
                )
            with anx_col:
                st.markdown(
                    anxiety_card_html(
                        result.get("gad7_band", "minimal"),
                        result.get("gad7_total", 0),
                    ),
                    unsafe_allow_html=True,
                )

            st.markdown("<div class='stack-gap'></div>", unsafe_allow_html=True)

            # Inline safety verdict only when not escalate (escalate shown full-width above)
            if result.get("safety_verdict") != "escalate":
                st.markdown("#### Safety verdict")
                st.markdown(
                    safety_verdict_html(
                        result.get("safety_verdict", "routine"),
                        result.get("deterministic_escalation", False),
                        result.get("critic_reasoning", ""),
                    ),
                    unsafe_allow_html=True,
                )

                if result.get("llm_verifier_flags"):
                    for flag in result["llm_verifier_flags"]:
                        st.caption(f"⚠ {flag}")

                st.markdown("<div class='stack-gap'></div>", unsafe_allow_html=True)

            st.markdown("#### DSM-5 signals detected")
            st.markdown(
                dsm_signals_html(result.get("dsm_differential") or []),
                unsafe_allow_html=True,
            )

        with col_right:
            st.markdown("#### Feature contributions")
            st.caption("Heuristic weights — exact SHAP values available after full model training (`make train`).")
            patient_input = result.get("patient_input", {})
            feat_names, feat_vals = phq9_item_contribution(patient_input)
            fig = feature_bar_chart(feat_names, feat_vals, "PHQ-9 item contributions (normalised)")
            st.pyplot(fig, use_container_width=True)

            st.markdown("---")
            st.markdown("#### Care plan suggestions")
            st.markdown(
                care_plan_html(result.get("care_plan_suggestions") or []),
                unsafe_allow_html=True,
            )

        # Scores summary row
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PHQ-9 total",          result.get("phq9_total", 0), help="0–27")
        m2.metric("GAD-7 total",           result.get("gad7_total", 0), help="0–21")
        m3.metric("Severity score",        f"{result.get('severity_score', 0.0):.3f}", help="Normalised 0–1")
        m4.metric("SI probability",        f"{result.get('suicidal_ideation_prob', 0.0):.3f}", help="From MentalBERT SI classifier")


# ── Tab 3: Literature ─────────────────────────────────────────────────────────

with tab_literature:
    result = st.session_state.get("result")

    if result is None:
        st.info("Submit an assessment to retrieve relevant literature.")
    else:
        citations = result.get("citations") or []
        dsm_context = result.get("dsm_context", "")
        pubmed_context = result.get("pubmed_context", "")
        dsm_differential = result.get("dsm_differential") or []

        if citations:
            st.markdown(f"#### {len(citations)} citations retrieved")
            for i, c in enumerate(citations, 1):
                score = c.get("score")
                score_badge = f"  `{score:.3f}`" if score is not None else ""
                label = f"{i}. {c.get('title') or c.get('source', 'Citation')}{score_badge}"
                with st.expander(label, expanded=(i <= 3)):
                    if c.get("pmid"):
                        st.caption(f"PMID: {c['pmid']}")
                    if c.get("disorder"):
                        st.caption(f"Disorder: {c['disorder']}")
                    if score is not None:
                        st.caption(f"Relevance score: {score:.3f}")
                    st.caption(f"Source: {c.get('source', 'N/A')}")
        else:
            st.markdown("""
<div style="background:oklch(97% 0 0); border:1px solid oklch(92.2% 0 0); border-radius:0.75rem;
            padding:14px 18px; font-size:0.85rem; color:oklch(37.1% 0 0); margin-bottom:16px;">
  <strong>RAG index not built.</strong> Run <code>make rag-index</code> to enable PubMed and DSM-5
  citation retrieval. The DSM context below is generated from heuristic fallback.
</div>
""", unsafe_allow_html=True)

        if dsm_differential:
            st.markdown("#### DSM-5 differential signals")
            for item in dsm_differential:
                disorder = item.get("disorder", "Unknown")
                confidence = item.get("confidence", "")
                rationale = item.get("rationale", "")
                with st.expander(f"{disorder}" + (f"  —  {confidence}" if confidence else ""), expanded=True):
                    if rationale:
                        st.markdown(rationale)

        if dsm_context and dsm_context not in ("", "RAG chain not available."):
            st.markdown("#### DSM-5 clinical context")
            st.markdown(dsm_context)

        if pubmed_context and pubmed_context not in ("", "RAG chain not available."):
            st.markdown("#### PubMed context")
            st.markdown(pubmed_context)

        if not citations and not dsm_context and not pubmed_context and not dsm_differential:
            st.info("No literature context was retrieved for this assessment.")
