"""
Feature engineering pipeline for the PHQ-9 / GAD-7 severity classifier.

Reads the Parquet produced by spark_jobs/synthetic_screening.py (or the
real-world equivalent) and returns train/val/test arrays ready for any
sklearn-compatible estimator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

PHQ9_ITEMS = [
    "phq1_anhedonia", "phq2_low_mood", "phq3_sleep", "phq4_fatigue",
    "phq5_appetite", "phq6_self_worth", "phq7_concentration",
    "phq8_psychomotor", "phq9_suicidal_ideation",
]
GAD7_ITEMS = [
    "gad1_nervous", "gad2_uncontrollable_worry", "gad3_excess_worry",
    "gad4_trouble_relaxing", "gad5_restless", "gad6_irritable", "gad7_fearful",
]
NUMERIC_COLS = PHQ9_ITEMS + GAD7_ITEMS + ["age", "phq9_total", "gad7_total"]
BINARY_COLS = ["prior_diagnosis", "on_medication"]
CATEGORICAL_COLS = ["gender", "education"]
DERIVED_COLS = [
    "somatic_cluster",      # phq3+phq4+phq5 (sleep, fatigue, appetite)
    "cognitive_cluster",    # phq6+phq7 (self-worth, concentration)
    "gad_physical_cluster", # gad4+gad5 (restless, trouble relaxing)
    "si_flag",              # phq9_suicidal_ideation > 0
    "phq_gad_ratio",        # phq9_total / (gad7_total + 1)
]

SEVERITY_BANDS = ["none", "mild", "moderate", "moderately_severe", "severe"]
BAND_TO_INT = {b: i for i, b in enumerate(SEVERITY_BANDS)}


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["somatic_cluster"] = df["phq3_sleep"] + df["phq4_fatigue"] + df["phq5_appetite"]
    df["cognitive_cluster"] = df["phq6_self_worth"] + df["phq7_concentration"]
    df["gad_physical_cluster"] = df["gad4_trouble_relaxing"] + df["gad5_restless"]
    df["si_flag"] = (df["phq9_suicidal_ideation"] > 0).astype(int)
    df["phq_gad_ratio"] = df["phq9_total"] / (df["gad7_total"] + 1)
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS + DERIVED_COLS),
            ("bin", "passthrough", BINARY_COLS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(str(path))


def prepare(
    df: pd.DataFrame,
    target_col: str = "phq9_band",
    split_col: str = "split",
) -> dict:
    """
    Return a dict with keys train/val/test, each containing X and y arrays,
    plus the fitted preprocessor, label encoder, and feature names.
    """
    df = add_derived(df)
    le = LabelEncoder()
    le.fit(SEVERITY_BANDS)

    preprocessor = build_preprocessor()

    splits = {}
    train_df = df[df[split_col] == "train"]
    preprocessor.fit(train_df)

    for split in ("train", "val", "test"):
        part = df[df[split_col] == split].copy()
        x = preprocessor.transform(part)
        y_label = part[target_col].values
        y = le.transform(y_label)
        y_ordinal = np.array([BAND_TO_INT[b] for b in y_label])
        splits[split] = {
            "X": x.astype(np.float32),
            "y": y,
            "y_ordinal": y_ordinal,
            "df": part,
        }

    feature_names = preprocessor.get_feature_names_out().tolist()

    return {
        "splits": splits,
        "preprocessor": preprocessor,
        "label_encoder": le,
        "feature_names": feature_names,
    }
