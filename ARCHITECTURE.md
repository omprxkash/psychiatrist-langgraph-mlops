# Architecture

Technical deep-dive for Psychiatrist — the agent pipeline, Safety-Critic design, fallback patterns, and key decisions. For the overview, start with [README.md](README.md).

---

## The pipeline

A linear LangGraph DAG with five nodes sharing one `AgentState` TypedDict:

```
START → screening → risk → dsm_literature → care_plan → safety_critic → END
```



There are **no conditional edges**. Every node runs on every request. The Safety-Critic always runs last and its verdict (`escalate` / `monitor` / `routine`) is the single output the FastAPI layer routes on.

State schema lives in [agents/state.py](agents/state.py). LangGraph merges each node's returned dict into the running state — agents read from state, return a partial dict of what they've changed.

---

## Agent responsibilities

### Screening agent — `agents/screening.py`
Validates input structure (PHQ-9 items clamped to 0–3, GAD-7 items clamped to 0–3), computes `phq9_total`, `gad7_total`, and derives `phq9_band` using standard cutoffs. Also flags `suicidal_ideation_positive` if PHQ-9 Q9 > 0 — this flag is read by the Safety-Critic's deterministic layer.

### Risk agent — `agents/risk.py`
Runs the severity classifier (XGBoost by default, falls back to rules if no checkpoint). Produces `severity_score` (0–1 continuous) and `active_symptoms`. Also runs the MentalBERT SI predictor on the clinical narrative to produce `suicidal_ideation_prob`. Falls back to regex keyword matching if MentalBERT isn't available.

### DSM/Literature agent — `agents/dsm_literature.py`
Queries the hybrid FAISS + BM25 retriever ([rag/retriever.py](rag/retriever.py)) over two indexes:
- `rag/indexes/dsm/` — DSM-5 paraphrased clinician summaries (never the raw APA text)
- `rag/indexes/pubmed/` — PubMed psychiatry abstracts

Returns `dsm_context`, `pubmed_context`, `dsm_differential`, and `citations`. Falls back to placeholder text if indexes aren't built.

### Care plan agent — `agents/care_plan.py`
Generates `care_plan_suggestions` from severity band, active symptoms, and retrieved literature. Uses Ollama (Llama 3.2) for LLM-generated suggestions; falls back to templated recommendations if Ollama is unavailable.

### Safety-Critic agent — `agents/safety_critic.py`
The most critical node. Two-layer architecture, deterministic-first:

**Layer 1 — `_check_deterministic` (always runs)**
Fires on any of:
- `suicidal_ideation_positive == True` (from Screening)
- `phq9_suicidal_ideation > 0` (Q9 raw score)
- Any `SI_KEYWORDS` regex match in the narrative — applied **sentence-by-sentence** with `_CONDITIONAL_FRAMES` exclusion to avoid false positives on reported speech (e.g. "the doctor asked me if I was suicidal")

If Layer 1 fires → `safety_verdict = "escalate"`, `deterministic_escalation = True`, Layer 2 is skipped.

**Layer 2 — `_run_llm_verifier` (only if Layer 1 did not fire)**
Checks the generated care plan for:
- Hallucinated symptoms not present in the narrative
- Unsupported diagnostic claims
- Inappropriate certainty in phrasing

Returns `llm_verifier_flags` and an advisory verdict. **The LLM cannot downgrade an escalation.** For PHQ-9 bands ≥ moderate, the LLM minimum is `monitor`.

`deterministic_escalation` is logged separately from `safety_verdict` so monitoring can track rule/LLM agreement over time.

---

## Dependency injection — the fallback pattern

Every agent accepts its external dependency as a constructor arg that defaults to `None`:

```python
class RiskAgent:
    def __init__(self, severity_clf=None, nlp_predictor=None):
        self.severity_clf = severity_clf   # None → rules-only PHQ-9 banding
        self.nlp_predictor = nlp_predictor # None → regex SI keyword fallback
```

`build_graph_from_env()` in [agents/graph.py](agents/graph.py) tries each dependency (load XGBoost checkpoint, connect to Ollama, load MentalBERT) and silently falls back. This is what makes the safety suite pass in CI without GPU / Ollama / trained checkpoints, and what makes the minimal quickstart work on a fresh laptop in under 10 minutes.

**Preserve this pattern when adding agents.** Never hard-import a model or external service at module level.

---

## Retriever — `rag/retriever.py`

Hybrid retrieval: dense (FAISS cosine) + sparse (BM25), combined with RRF (Reciprocal Rank Fusion). Query hits both indexes, results are merged and re-ranked. The two indexes are intentionally separate — DSM summaries have different retrieval characteristics than PubMed abstracts, and keeping them split allows per-source relevance tuning.

Entry point: `HybridRetriever.from_index_dirs({"dsm": ..., "pubmed": ...})`.

---

## Serving layer

| Surface | File | Notes |
|---|---|---|
| Streamlit UI | [serving/app.py](serving/app.py) | PHQ-9 + GAD-7 intake, results tabs, audit log write |
| UI components | [serving/components.py](serving/components.py) | severity card, safety verdict, DSM signals, care plan, feature chart |
| FastAPI service | [serving/api.py](serving/api.py) | POST `/analyze`, GET `/health`, GET `/ready`, GET `/docs`. Rate-limited: `RATE_LIMIT_RPM` req/min per IP (default 10). CORS enabled. |

---

## MLOps

| Tool | Status | Config |
|---|---|---|
| MLflow | Dependency installed | No remote server configured — would need `MLFLOW_TRACKING_URI` |
| DVC | Dependency installed | No `.dvc/` remote configured |
| Evidently | [monitoring/drift_report.py](monitoring/drift_report.py) | Runs locally on `data/processed/` output |
| Audit log | [monitoring/audit_log.py](monitoring/audit_log.py) | Append-only JSONL per-request log. No PII — feature vectors and verdicts only. |
| Safety audit | [monitoring/safety_audit.py](monitoring/safety_audit.py) | Tracks rule/LLM agreement over time |

---

## Stack decisions (locked)

These were chosen deliberately — don't replace without a concrete reason:

- **LangGraph** not vanilla LangChain agents — graph topology is explicit and testable; interviewers ask about it specifically.
- **Ollama + Llama-3.2-3B / Phi-3-mini** locally. No paid LLM APIs in any code path.
- **MentalBERT** (`mental/mental-bert-base-uncased`), int8-quantized. Outperforms generic 7B models for SI detection on a CPU laptop because it's domain-pretrained on mental health text.
- **One file per agent.** Each LangGraph node is its own callable class. Easier to unit-test, easier to mock, easier to swap. Don't merge agents into a mega-orchestrator.
- **Synthetic data generator is pandas, not PySpark, by default.** The Spark job is the production-shaped path; for local dev nobody wants to install a JDK to generate fake screening data.
- **Deterministic rules before LLM.** The LLM can add nuance (negation, grief context) but it can never lower a deterministic escalation. This is why the safety suite stays at 100% recall even with the LLM disabled.
- Python 3.10–3.12. `requires-python = ">=3.10,<3.13"` in pyproject.toml.

---

## Safety contract

Read [SAFETY.md](SAFETY.md) before touching `agents/safety_critic.py` or `tests/safety/`.

The suicide-risk regression test set in [tests/safety/suicide_risk_cases.py](tests/safety/suicide_risk_cases.py) is a **durable contract**. New cases can be added; existing `expected_verdict="escalate"` cases must not be relaxed without the SAFETY.md review process (test-set update + SAFETY.md note + a second reviewer). Any PR that lowers recall must be rejected.
