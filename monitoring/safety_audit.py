"""
Safety escalation audit script.

Reads the prediction audit log and summarises cases where the Safety-Critic
escalated.  Useful for Monday-morning review of the previous week's edge cases —
the kind of thing you'd run to spot systematic failure modes before they become
clinical incidents.

Usage:
    python monitoring/safety_audit.py
    python monitoring/safety_audit.py --last-n 200 --verdict escalate
    python monitoring/safety_audit.py --export escalations.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def main(last_n: int | None, verdict_filter: str | None, export: str | None) -> None:
    from monitoring.audit_log import LOG_PATH, load_log

    records = load_log(last_n)
    if not records:
        print(f"No records in audit log ({LOG_PATH}).")
        sys.exit(0)

    # Filter
    if verdict_filter:
        records = [r for r in records if r.get("safety_verdict") == verdict_filter]

    escalated = [r for r in records if r.get("safety_verdict") == "escalate"]
    monitored  = [r for r in records if r.get("safety_verdict") == "monitor"]
    routine    = [r for r in records if r.get("safety_verdict") == "routine"]
    determ     = [r for r in records if r.get("deterministic_escalation")]

    total = len(records)
    print("=" * 65)
    print("  Psychiatrist  —  Safety Audit Summary")
    print("=" * 65)
    print(f"  Records reviewed : {total}")
    print(f"  Escalate         : {len(escalated):>5}  ({len(escalated)/total:.1%})")
    print(f"  Monitor          : {len(monitored):>5}  ({len(monitored)/total:.1%})")
    print(f"  Routine          : {len(routine):>5}  ({len(routine)/total:.1%})")
    print(f"  Deterministic escalation triggered : {len(determ)}")
    print("=" * 65)

    if not records:
        print("\nNo records to display after filtering.")
        return

    # Show individual escalations
    show = [r for r in records if r.get("safety_verdict") in ("escalate", "monitor")]
    if show:
        print(f"\nEscalated / Monitor cases ({len(show)}):\n")
        header = f"  {'Timestamp':<20} {'PHQ-9':>5} {'GAD-7':>5}  {'Band':<18}  {'Verdict':<8}  {'SI_prob':>7}  {'Determ':>6}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for r in show:
            ts      = r["timestamp"][:19].replace("T", " ")
            phq     = r.get("phq9_total", 0)
            gad     = r.get("gad7_total", 0)
            band    = r.get("phq9_band", "?")
            verdict = r.get("safety_verdict", "?")
            si_prob = r.get("suicidal_ideation_prob", 0.0)
            determ  = "YES" if r.get("deterministic_escalation") else "no"
            print(f"  {ts:<20} {phq:>5} {gad:>5}  {band:<18}  {verdict:<8}  {si_prob:>7.3f}  {determ:>6}")
    else:
        print("\nNo escalate/monitor cases in this window — looks clean.")

    # Optionally export to CSV
    if export:
        export_path = Path(export)
        if records:
            keys = list(records[0].keys())
            with export_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(records)
            print(f"\nExported {len(records)} records to {export_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Review safety escalations in the audit log")
    parser.add_argument("--last-n",  type=int,  default=None,
                        help="Limit to last N records (default: all)")
    parser.add_argument("--verdict", type=str,  default=None,
                        choices=["escalate", "monitor", "routine"],
                        help="Filter by safety verdict")
    parser.add_argument("--export",  type=str,  default=None,
                        help="Export filtered records to CSV")
    args = parser.parse_args()
    main(args.last_n, args.verdict, args.export)
