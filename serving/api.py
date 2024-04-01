"""
FastAPI serving layer for Psychiatrist.

Endpoints:
  POST /analyze   — main inference endpoint
  GET  /health    — liveness check
  GET  /ready     — readiness check (graph loaded)
  GET  /docs      — auto-generated OpenAPI docs (built-in)

Rate limiting: 10 requests/minute per IP (configurable via env RATE_LIMIT_RPM).
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.base import BaseHTTPMiddleware

# ── Pydantic models ────────────────────────────────────────────────────────────

class PHQ9(BaseModel):
    phq1_anhedonia: int = Field(ge=0, le=3)
    phq2_low_mood: int = Field(ge=0, le=3)
    phq3_sleep: int = Field(ge=0, le=3)
    phq4_fatigue: int = Field(ge=0, le=3)
    phq5_appetite: int = Field(ge=0, le=3)
    phq6_self_worth: int = Field(ge=0, le=3)
    phq7_concentration: int = Field(ge=0, le=3)
    phq8_psychomotor: int = Field(ge=0, le=3)
    phq9_suicidal_ideation: int = Field(ge=0, le=3)


class GAD7(BaseModel):
    gad1_nervous: int = Field(ge=0, le=3)
    gad2_uncontrollable_worry: int = Field(ge=0, le=3)
    gad3_excess_worry: int = Field(ge=0, le=3)
    gad4_trouble_relaxing: int = Field(ge=0, le=3)
    gad5_restless: int = Field(ge=0, le=3)
    gad6_irritable: int = Field(ge=0, le=3)
    gad7_fearful: int = Field(ge=0, le=3)


class AnalyzeRequest(BaseModel):
    phq9: PHQ9
    gad7: GAD7
    narrative: str = Field(min_length=1, max_length=4000)
    age: int = Field(ge=18, le=100)
    gender: str = Field(default="unknown", max_length=20)

    @field_validator("narrative")
    @classmethod
    def narrative_not_placeholder(cls, v: str) -> str:
        if v.strip().lower() in ("test", "n/a", "none", ""):
            raise ValueError("narrative must be a meaningful clinical text")
        return v


class Citation(BaseModel):
    source: str
    pmid: str | None = None
    disorder: str | None = None
    title: str | None = None
    score: float | None = None


class AnalyzeResponse(BaseModel):
    phq9_total: int
    gad7_total: int
    phq9_band: str
    gad7_band: str
    severity_score: float
    suicidal_ideation_prob: float
    active_symptoms: list[str]
    dsm_differential: list[str]
    citations: list[Citation]
    care_plan_suggestions: list[str]
    safety_verdict: str
    critic_reasoning: str
    deterministic_escalation: bool
    llm_verifier_flags: list[str]
    processing_time_ms: float

    model_config = {"protected_namespaces": ()}


# ── App lifecycle ──────────────────────────────────────────────────────────────

_graph: Any = None
_graph_ready = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph, _graph_ready
    try:
        from agents.graph import build_graph_from_env
        _graph = build_graph_from_env()
        _graph_ready = True
        print("Psychiatrist graph loaded successfully.")
    except Exception as e:
        print(f"WARNING: Graph failed to load ({e}). /analyze will return 503.")
        _graph_ready = False
    yield
    _graph_ready = False


app = FastAPI(
    title="Psychiatrist",
    description=(
        "Agentic mental-health triage and clinical decision support. "
        "NOT a medical device. See /safety for full disclaimer."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class _SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


app.add_middleware(_SecurityHeadersMiddleware)


# ── Simple in-memory rate limiter ─────────────────────────────────────────────

RATE_LIMIT_RPM = int(os.environ.get("RATE_LIMIT_RPM", "10"))
_rate_store: dict[str, list[float]] = defaultdict(list)
_RATE_STORE_MAX_IPS = 10_000


def _check_rate_limit(ip: str):
    now = time.time()
    window = 60.0

    # Evict stale IPs when the store grows too large to prevent unbounded memory growth.
    if len(_rate_store) > _RATE_STORE_MAX_IPS:
        stale = [k for k, ts in _rate_store.items() if not any(now - t < window for t in ts)]
        for k in stale:
            del _rate_store[k]

    requests = [t for t in _rate_store[ip] if now - t < window]
    if len(requests) >= RATE_LIMIT_RPM:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit: {RATE_LIMIT_RPM} requests/minute exceeded.",
        )
    requests.append(now)
    _rate_store[ip] = requests


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if not _graph_ready:
        raise HTTPException(status_code=503, detail="Graph not loaded yet.")
    return {"status": "ready"}


@app.get("/safety")
def safety_disclaimer():
    return {
        "disclaimer": (
            "Psychiatrist is a portfolio engineering project. "
            "It is NOT a medical device, NOT clinically validated, "
            "and NOT a substitute for a licensed mental health professional. "
            "If you or someone you know is in crisis, call a helpline: "
            "India: iCall +91 9152987821, Vandrevala 1860-2662-345, KIRAN 1800-599-0019."
        )
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(client_ip)

    if not _graph_ready or _graph is None:
        raise HTTPException(status_code=503, detail="Service not ready. Try again shortly.")

    patient_input = {
        **req.phq9.model_dump(),
        **req.gad7.model_dump(),
        "narrative": req.narrative,
        "age": req.age,
        "gender": req.gender,
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
        "report_ready": False, "error": None,
    }

    t0 = time.perf_counter()
    try:
        final_state = _graph.invoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Inference error. Check server logs.") from e
    elapsed_ms = (time.perf_counter() - t0) * 1000

    citations = [Citation(**c) for c in (final_state.get("citations") or [])]

    return AnalyzeResponse(
        phq9_total=final_state.get("phq9_total", 0),
        gad7_total=final_state.get("gad7_total", 0),
        phq9_band=final_state.get("phq9_band", "none"),
        gad7_band=final_state.get("gad7_band", "minimal"),
        severity_score=round(final_state.get("severity_score", 0.0), 4),
        suicidal_ideation_prob=round(final_state.get("suicidal_ideation_prob", 0.0), 4),
        active_symptoms=final_state.get("active_symptoms") or [],
        dsm_differential=final_state.get("dsm_differential") or [],
        citations=citations,
        care_plan_suggestions=final_state.get("care_plan_suggestions") or [],
        safety_verdict=final_state.get("safety_verdict", "routine"),
        critic_reasoning=final_state.get("critic_reasoning", ""),
        deterministic_escalation=final_state.get("deterministic_escalation", False),
        llm_verifier_flags=final_state.get("llm_verifier_flags") or [],
        processing_time_ms=round(elapsed_ms, 1),
    )
