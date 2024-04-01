"""
Append-only prediction audit log.

Every call to log_prediction() appends one JSON line to
monitoring/predictions.jsonl.  The file is the source of truth for
drift_report.py and safety_audit.py.

Design notes:
- No PII stored — only feature vectors and model outputs.
- Append-only: never overwrite or delete lines, so the audit trail is tamper-evident.
- Tiny footprint: one JSON object per line, no database required.
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path(__file__).parent / "predictions.jsonl"


def log_prediction(patient_input: dict, result: dict, model_version: str = "fallback") -> None:
    """Append one prediction record to the audit log."""
    record = {
        "log_id":       str(uuid.uuid4()),
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        # Feature vector (all PHQ-9 and GAD-7 items + demographics)
        "phq1_anhedonia":            patient_input.get("phq1_anhedonia", 0),
        "phq2_low_mood":             patient_input.get("phq2_low_mood", 0),
        "phq3_sleep":                patient_input.get("phq3_sleep", 0),
        "phq4_fatigue":              patient_input.get("phq4_fatigue", 0),
        "phq5_appetite":             patient_input.get("phq5_appetite", 0),
        "phq6_self_worth":           patient_input.get("phq6_self_worth", 0),
        "phq7_concentration":        patient_input.get("phq7_concentration", 0),
        "phq8_psychomotor":          patient_input.get("phq8_psychomotor", 0),
        "phq9_suicidal_ideation":    patient_input.get("phq9_suicidal_ideation", 0),
        "gad1_nervous":              patient_input.get("gad1_nervous", 0),
        "gad2_uncontrollable_worry": patient_input.get("gad2_uncontrollable_worry", 0),
        "gad3_excess_worry":         patient_input.get("gad3_excess_worry", 0),
        "gad4_trouble_relaxing":     patient_input.get("gad4_trouble_relaxing", 0),
        "gad5_restless":             patient_input.get("gad5_restless", 0),
        "gad6_irritable":            patient_input.get("gad6_irritable", 0),
        "gad7_fearful":              patient_input.get("gad7_fearful", 0),
        "age":                       patient_input.get("age", 0),
        "gender":                    patient_input.get("gender", "unknown"),
        # Model outputs
        "phq9_total":             result.get("phq9_total", 0),
        "gad7_total":             result.get("gad7_total", 0),
        "phq9_band":              result.get("phq9_band", "none"),
        "severity_score":         result.get("severity_score", 0.0),
        "suicidal_ideation_prob": result.get("suicidal_ideation_prob", 0.0),
        "safety_verdict":         result.get("safety_verdict", "routine"),
        "deterministic_escalation": result.get("deterministic_escalation", False),
    }

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_log(last_n: int | None = None) -> list[dict]:
    """Return audit log records, optionally limited to the most recent N."""
    if not LOG_PATH.exists():
        return []
    with LOG_PATH.open(encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    records = [json.loads(line) for line in lines]
    if last_n is not None:
        records = records[-last_n:]
    return records


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect the prediction audit log")
    parser.add_argument("--last-n", type=int, default=20, help="Show last N records")
    args = parser.parse_args()

    records = load_log(args.last_n)
    if not records:
        print(f"No records found at {LOG_PATH}")
        sys.exit(0)

    print(f"Audit log: {LOG_PATH}  ({len(load_log())} total records)")
    print(f"Showing last {len(records)}:\n")
    for r in records:
        ts = r["timestamp"][:19].replace("T", " ")
        verdict = r["safety_verdict"].upper()
        band = r["phq9_band"]
        phq = r["phq9_total"]
        gad = r["gad7_total"]
        esc = "  ← ESCALATED" if r.get("deterministic_escalation") else ""
        print(f"  {ts}  PHQ-9={phq:>2}  GAD-7={gad:>2}  band={band:<18}  verdict={verdict}{esc}")
