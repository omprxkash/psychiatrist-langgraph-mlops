"""
Synthetic patient screening data generator.

Produces ~50 k rows of plausible PHQ-9 / GAD-7 screening records with
derived features, severity labels, and a train/val/test split — all saved
as Parquet under data/processed/.

This replaces the PySpark ETL for local dev and CI.  The spark_jobs/ version
is still present for production-scale or cluster runs.  Distributions are
calibrated against published PHQ-9 prevalence tables (Kroenke et al. 2001;
Manea et al. 2012).
"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

# ── Clinical thresholds ────────────────────────────────────────────────────────

PHQ9_BANDS = [
    (0,  4,  "none"),
    (5,  9,  "mild"),
    (10, 14, "moderate"),
    (15, 19, "moderately_severe"),
    (20, 27, "severe"),
]

GAD7_BANDS = [
    (0,  4,  "minimal"),
    (5,  9,  "mild"),
    (10, 14, "moderate"),
    (15, 21, "severe"),
]

# ── Narrative templates by severity band ──────────────────────────────────────

NARRATIVES = {
    "none": [
        "Patient reports generally stable mood with no significant concerns. Sleep and appetite are normal. Denies any suicidal ideation.",
        "No active psychiatric symptoms. Patient is managing daily responsibilities well and engaging in social activities.",
        "Feeling well overall. Minor work stress noted but coping adequately. No sleep disturbance or changes in appetite.",
        "Patient denies depressive or anxious symptoms. Engaged and cooperative during assessment. GAD and PHQ scores within normal range.",
    ],
    "mild": [
        "Patient describes intermittent low mood over the past two weeks, particularly in the evenings. Sleep is mildly disrupted. Appetite slightly reduced. Denies SI.",
        "Reports mild anxiety about work deadlines and some fatigue. Concentration mildly affected. No functional impairment noted at this time.",
        "Occasional feelings of hopelessness described, not pervasive. Some difficulty relaxing. Patient attributes symptoms to a recent life stressor.",
        "Mild anhedonia reported — less interest in hobbies that were previously enjoyable. No SI. Supportive network in place.",
    ],
    "moderate": [
        "Patient presents with persistent low mood for approximately three weeks. Sleep is significantly disrupted with early morning awakening. Appetite markedly reduced. Concentration poor. Denies active SI but acknowledges passive death wishes on two occasions.",
        "Notable anxiety with frequent worry episodes that the patient cannot control. Restlessness affecting work performance. Some depressive overlay with reduced motivation and energy.",
        "Depressed mood most of the day, more days than not. Feelings of worthlessness present. Psychomotor slowing observed clinically. Denies SI currently.",
        "Reports significant fatigue and cognitive slowing. Difficulty completing tasks at work. Partner noted changes in mood and withdrawal from social activities.",
    ],
    "moderately_severe": [
        "Patient reports severe depressive episode with near-constant low mood, marked anhedonia, and significant functional decline. Sleep reduced to 3–4 hours per night. Suicidal ideation present — passive, with no plan or intent at this time.",
        "Presenting with severe anxiety, panic attacks occurring 2–3 times per week, and marked avoidance behaviour. Comorbid depressive symptoms. PHQ-9 indicates moderately severe depression.",
        "Patient has been unable to work for the past week due to depression. Reports feeling like a burden to family. SI present on most days but denies active plan. Urgent review required.",
        "Significant psychomotor agitation. Racing thoughts, sleep onset insomnia, and poor appetite leading to weight loss. Patient has history of a prior depressive episode.",
    ],
    "severe": [
        "Patient presents in significant distress with severe depression and active suicidal ideation with a loosely formed plan. Immediate psychiatric review arranged. Crisis resources provided.",
        "Severe depressive episode with psychotic features — patient reports hearing critical voices. Marked functional impairment. PHQ-9 score 24. Safety assessment initiated.",
        "Patient has been self-isolating for two weeks. Reports persistent suicidal ideation with intent but no specific plan. Admitted voluntarily for observation. Collateral history obtained from family.",
        "Severe anxiety and depression with significant dissociative episodes. Daily functioning severely compromised. Patient is currently not safe to be left alone according to family report.",
    ],
}


def _band_from_total(total: int, bands: list) -> str:
    for lo, hi, label in bands:
        if lo <= total <= hi:
            return label
    return bands[-1][2]


def _generate_items(
    n: int, n_items: int, latent_total: np.ndarray, max_total: int, rng: np.random.Generator
) -> np.ndarray:
    """Distribute a latent total score across n_items integer scores in [0,3]."""
    items = np.zeros((n, n_items), dtype=np.int8)
    for i in range(n):
        target = int(np.clip(latent_total[i], 0, max_total))
        if target == 0:
            continue
        remaining = target
        order = rng.permutation(n_items)
        for j in order:
            val = min(3, remaining)
            val = max(0, val - int(rng.integers(0, 2)))
            items[i, order[j]] = val
            remaining -= val
            if remaining <= 0:
                break
        deficit = target - items[i].sum()
        if deficit > 0:
            slot = rng.integers(n_items)
            items[i, slot] = min(3, items[i, slot] + deficit)
    return items


def generate(n: int = 50_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Severity group: ~40% none, 25% mild, 20% moderate, 10% mod-severe, 5% severe
    severity_weights = [0.40, 0.25, 0.20, 0.10, 0.05]
    severity_midpoints = [2.0, 7.0, 12.0, 17.0, 23.0]
    severity_stds      = [1.5, 2.0, 2.0,  2.0,  2.0]

    group = rng.choice(5, size=n, p=severity_weights)
    phq9_latent = np.array([
        rng.normal(severity_midpoints[g], severity_stds[g]) for g in group
    ])
    # Anxiety correlates with depression (r ≈ 0.65 in clinical populations).
    # Noise SD must be comparable to the signal SD to avoid over-correlation.
    noise = rng.normal(0, 3, size=n)
    gad7_latent = 0.6 * phq9_latent * (21 / 27) + noise

    phq9_items_raw = _generate_items(n, 9, phq9_latent, 27, rng)
    gad7_items_raw = _generate_items(n, 7, gad7_latent, 21, rng)

    # PHQ-9 Q9 (suicidal ideation) is sparse — reset it then re-add realistically.
    phq9_items_raw[:, 8] = 0
    for i in range(n):
        if group[i] >= 3:         # moderately_severe or severe
            if rng.random() < 0.45:
                phq9_items_raw[i, 8] = int(rng.choice([1, 2, 3], p=[0.5, 0.3, 0.2]))
        elif group[i] == 2 and rng.random() < 0.12:       # moderate
            phq9_items_raw[i, 8] = 1

    phq9_total = phq9_items_raw.sum(axis=1).astype(int)
    gad7_total = np.clip(gad7_items_raw.sum(axis=1), 0, 21).astype(int)

    ages = rng.integers(18, 81, size=n)
    genders = rng.choice(
        ["female", "male", "non-binary", "prefer_not_to_say"],
        size=n,
        p=[0.48, 0.45, 0.04, 0.03],
    )
    education = rng.choice(
        ["secondary", "undergraduate", "postgraduate", "no_formal"],
        size=n,
        p=[0.35, 0.40, 0.18, 0.07],
    )
    prior_diagnosis = (rng.random(n) < 0.28).astype(int)
    on_medication   = (prior_diagnosis.astype(bool) & (rng.random(n) < 0.55)).astype(int)

    phq9_band = [_band_from_total(t, PHQ9_BANDS) for t in phq9_total]
    gad7_band = [_band_from_total(t, GAD7_BANDS) for t in gad7_total]

    band_keys = ["none", "mild", "moderate", "moderately_severe", "severe"]
    narratives = [
        rng.choice(NARRATIVES[band_keys[g]]) for g in group
    ]

    split_arr = rng.choice(
        ["train", "val", "test"], size=n, p=[0.70, 0.15, 0.15]
    )

    df = pd.DataFrame({
        "session_id": [str(uuid.uuid4()) for _ in range(n)],
        # PHQ-9 items
        "phq1_anhedonia":          phq9_items_raw[:, 0],
        "phq2_low_mood":           phq9_items_raw[:, 1],
        "phq3_sleep":              phq9_items_raw[:, 2],
        "phq4_fatigue":            phq9_items_raw[:, 3],
        "phq5_appetite":           phq9_items_raw[:, 4],
        "phq6_self_worth":         phq9_items_raw[:, 5],
        "phq7_concentration":      phq9_items_raw[:, 6],
        "phq8_psychomotor":        phq9_items_raw[:, 7],
        "phq9_suicidal_ideation":  phq9_items_raw[:, 8],
        # GAD-7 items
        "gad1_nervous":            gad7_items_raw[:, 0],
        "gad2_uncontrollable_worry": gad7_items_raw[:, 1],
        "gad3_excess_worry":       gad7_items_raw[:, 2],
        "gad4_trouble_relaxing":   gad7_items_raw[:, 3],
        "gad5_restless":           gad7_items_raw[:, 4],
        "gad6_irritable":          gad7_items_raw[:, 5],
        "gad7_fearful":            gad7_items_raw[:, 6],
        # Totals
        "phq9_total":        phq9_total,
        "gad7_total":        gad7_total,
        # Demographics
        "age":               ages,
        "gender":            genders,
        "education":         education,
        "prior_diagnosis":   prior_diagnosis,
        "on_medication":     on_medication,
        # Narrative
        "narrative":         narratives,
        # Labels
        "phq9_band":         phq9_band,
        "gad7_band":         gad7_band,
        "severity_group":    group,
        # Split
        "split":             split_arr,
    })

    return df


def main(n: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {n:,} synthetic patient records…")
    df = generate(n)

    # Raw screening Parquet
    screening_path = out_dir / "synthetic_screening.parquet"
    df.to_parquet(screening_path, index=False)
    print(f"  Saved: {screening_path}  ({len(df):,} rows)")

    # Feature-engineered Parquet (adds derived cols used by severity classifier)
    from models.severity.features import add_derived
    df_feat = add_derived(df)
    features_path = out_dir / "features.parquet"
    df_feat.to_parquet(features_path, index=False)
    print(f"  Saved: {features_path}  ({len(df_feat):,} rows, {len(df_feat.columns)} cols)")

    # Quick sanity: severity distribution
    print("\nSeverity distribution (phq9_band):")
    print(df["phq9_band"].value_counts().sort_index().to_string())
    print(f"\nSI flag (PHQ-9 Q9 > 0): {(df['phq9_suicidal_ideation'] > 0).sum()} / {n} "
          f"({(df['phq9_suicidal_ideation'] > 0).mean():.1%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic Psychiatrist data")
    parser.add_argument("--n", type=int, default=50_000, help="Number of records")
    parser.add_argument("--out", type=str, default="data/processed", help="Output directory")
    args = parser.parse_args()
    main(args.n, Path(args.out))
