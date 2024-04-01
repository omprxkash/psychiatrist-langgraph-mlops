"""
Model evaluation: SHAP explanations, calibration curves, and subgroup fairness slices.

All outputs are saved as MLflow artifacts so they appear in the experiment UI.
"""

from __future__ import annotations

import io
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    cohen_kappa_score,
    f1_score,
)

# ── SHAP ──────────────────────────────────────────────────────────────────────

def compute_shap(
    clf: Any,
    x: np.ndarray,
    feature_names: list[str],
    max_samples: int = 2000,
) -> Any:
    try:
        import shap
    except ImportError as e:
        raise ImportError("shap is required: pip install shap") from e

    x_sample = x[:max_samples]
    explainer = shap.TreeExplainer(clf) if hasattr(clf, "get_booster") else shap.Explainer(clf, x_sample)
    return explainer(x_sample)


def log_shap_plots(shap_values, feature_names: list[str], class_labels: list[str]):
    """Log SHAP beeswarm + bar plots for each class as MLflow artifacts."""
    try:
        import shap
    except ImportError:
        return

    for cls_idx, cls_name in enumerate(class_labels):
        vals = shap_values[..., cls_idx] if shap_values.values.ndim == 3 else shap_values

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        plt.sca(axes[0])
        shap.plots.beeswarm(vals, max_display=15, show=False)
        axes[0].set_title(f"SHAP beeswarm — {cls_name}")

        plt.sca(axes[1])
        shap.plots.bar(vals, max_display=15, show=False)
        axes[1].set_title(f"SHAP feature importance — {cls_name}")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        mlflow.log_image(buf, key=f"shap_{cls_name}.png")
        plt.close(fig)


# ── Calibration ───────────────────────────────────────────────────────────────

def log_calibration_curves(
    clf: Any,
    x: np.ndarray,
    y: np.ndarray,
    class_labels: list[str],
):
    """One-vs-rest calibration curve per severity band."""
    y_proba = clf.predict_proba(x)
    n_classes = y_proba.shape[1]

    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 4), sharey=True)
    if n_classes == 1:
        axes = [axes]

    for cls_idx, cls_name in enumerate(class_labels[:n_classes]):
        y_bin = (y == cls_idx).astype(int)
        prob = y_proba[:, cls_idx]
        try:
            CalibrationDisplay.from_predictions(
                y_bin, prob, n_bins=10, ax=axes[cls_idx], name=cls_name
            )
        except Exception:
            axes[cls_idx].set_title(f"{cls_name} — calibration unavailable")
        axes[cls_idx].set_title(f"Calibration — {cls_name}")

    fig.suptitle("Calibration curves (one-vs-rest per severity band)", y=1.02)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    mlflow.log_image(buf, key="calibration_curves.png")
    plt.close(fig)


# ── Confusion matrix ──────────────────────────────────────────────────────────

def log_confusion_matrix(y_true, y_pred, class_labels: list[str]):
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=class_labels, ax=ax, cmap="Blues", normalize="true"
    )
    ax.set_title("Normalised confusion matrix (row = true label)")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    mlflow.log_image(buf, key="confusion_matrix.png")
    plt.close(fig)


# ── Subgroup fairness ─────────────────────────────────────────────────────────

FAIRNESS_SLICES = {
    "gender": ["female", "male", "nonbinary"],
    "age_band": ["18-29", "30-44", "45-59", "60+"],
}


def _age_band(age: int) -> str:
    if age < 30:
        return "18-29"
    if age < 45:
        return "30-44"
    if age < 60:
        return "45-59"
    return "60+"


def compute_fairness_slices(
    clf: Any,
    x: np.ndarray,
    y_true: np.ndarray,
    df_meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute weighted F1 and quadratic kappa per subgroup slice.

    Returns a DataFrame with columns: slice_col, slice_val, n, weighted_f1, quadratic_kappa.
    """
    df_meta = df_meta.copy()
    if "age" in df_meta.columns:
        df_meta["age_band"] = df_meta["age"].apply(_age_band)

    y_pred = clf.predict(x)
    records = []

    for col, vals in FAIRNESS_SLICES.items():
        if col not in df_meta.columns:
            continue
        for val in vals:
            mask = (df_meta[col] == val).values
            if mask.sum() < 10:
                continue
            y_t = y_true[mask]
            y_p = y_pred[mask]
            records.append(
                {
                    "slice_col": col,
                    "slice_val": val,
                    "n": int(mask.sum()),
                    "weighted_f1": f1_score(y_t, y_p, average="weighted", zero_division=0),
                    "quadratic_kappa": cohen_kappa_score(y_t, y_p, weights="quadratic")
                    if len(np.unique(y_t)) > 1
                    else float("nan"),
                }
            )

    return pd.DataFrame(records)


def log_fairness_report(fairness_df: pd.DataFrame):
    """Log fairness slice table as a CSV artifact and compute max disparity metrics."""
    buf = io.StringIO()
    fairness_df.to_csv(buf, index=False)
    buf.seek(0)
    mlflow.log_text(buf.getvalue(), artifact_file="fairness_slices.csv")

    for col in FAIRNESS_SLICES:
        subset = fairness_df[fairness_df["slice_col"] == col]
        if subset.empty:
            continue
        disparity = subset["weighted_f1"].max() - subset["weighted_f1"].min()
        mlflow.log_metric(f"fairness_f1_disparity_{col}", round(disparity, 4))

    _log_fairness_chart(fairness_df)


def _log_fairness_chart(fairness_df: pd.DataFrame):
    n_cols = fairness_df["slice_col"].nunique()
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4), sharey=False)
    if n_cols == 1:
        axes = [axes]

    for ax, (col, group) in zip(axes, fairness_df.groupby("slice_col"), strict=False):
        ax.barh(group["slice_val"], group["weighted_f1"], color="steelblue")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Weighted F1")
        ax.set_title(f"Fairness by {col}")
        for _, row in group.iterrows():
            ax.text(row["weighted_f1"] + 0.01, row["slice_val"], f'{row["weighted_f1"]:.3f}', va="center")

    fig.suptitle("Subgroup fairness — weighted F1 per slice")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    mlflow.log_image(buf, key="fairness_slices.png")
    plt.close(fig)
