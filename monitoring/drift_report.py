"""
Feature drift report using Evidently.

Compares the distribution of recent predictions (from the audit log) against
the reference dataset (data/processed/synthetic_screening.parquet).

Why Evidently over a custom KL-divergence implementation: it handles mixed
numeric/categorical columns cleanly and produces a shareable HTML report
that non-technical stakeholders can open in a browser.

The most important column to watch here is phq9_suicidal_ideation (PHQ-9 Q9).
If real-world SI prevalence drifts above what the training data showed, the
model's calibration for high-risk cases may degrade — this is the signal most
worth acting on.

Usage:
    python monitoring/drift_report.py
    python monitoring/drift_report.py --last-n 500 --ref data/processed/features.parquet
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPORT_DIR = Path(__file__).parent / "reports"
DEFAULT_REF = Path("data/processed/synthetic_screening.parquet")

NUMERIC_COLS = [
    "phq1_anhedonia", "phq2_low_mood", "phq3_sleep", "phq4_fatigue",
    "phq5_appetite", "phq6_self_worth", "phq7_concentration",
    "phq8_psychomotor", "phq9_suicidal_ideation",
    "gad1_nervous", "gad2_uncontrollable_worry", "gad3_excess_worry",
    "gad4_trouble_relaxing", "gad5_restless", "gad6_irritable", "gad7_fearful",
    "phq9_total", "gad7_total", "age",
]
CATEGORICAL_COLS = ["gender", "phq9_band"]


def build_report(reference: pd.DataFrame, current: pd.DataFrame) -> str:
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
    except ImportError:
        print("evidently not installed. Run: pip install evidently", file=sys.stderr)
        sys.exit(1)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / f"drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    report.save_html(str(out_path))
    return str(out_path)


def _load_reference(ref_path: Path, cols: list[str]) -> pd.DataFrame:
    if not ref_path.exists():
        print(f"Reference data not found at {ref_path}. Run `make data` first.", file=sys.stderr)
        sys.exit(1)
    df = pd.read_parquet(str(ref_path))
    available = [c for c in cols if c in df.columns]
    return df[available].copy()


def _load_current(last_n: int | None) -> pd.DataFrame:
    from monitoring.audit_log import load_log
    records = load_log(last_n)
    if not records:
        print("Audit log is empty — run some predictions via the UI first.", file=sys.stderr)
        sys.exit(1)
    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} prediction records from audit log.")
    return df


def main(last_n: int | None, ref_path: Path) -> None:
    cols = NUMERIC_COLS + CATEGORICAL_COLS
    reference = _load_reference(ref_path, cols)
    current   = _load_current(last_n)

    # Align columns — keep only what's in both
    shared = [c for c in cols if c in reference.columns and c in current.columns]
    reference = reference[shared]
    current   = current[shared]

    print(f"Reference: {len(reference)} rows  |  Current: {len(current)} rows")
    print(f"Columns compared: {shared}")

    out_path = build_report(reference, current)
    print(f"\nDrift report saved: {out_path}")
    print("Open in a browser to review column-level drift statistics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Evidently drift report")
    parser.add_argument("--last-n", type=int, default=None,
                        help="Limit audit log to last N records (default: all)")
    parser.add_argument("--ref", type=str, default=str(DEFAULT_REF),
                        help="Path to reference Parquet file")
    args = parser.parse_args()
    main(args.last_n, Path(args.ref))
