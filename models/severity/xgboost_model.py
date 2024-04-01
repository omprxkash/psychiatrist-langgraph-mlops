"""XGBoost severity classifier with MLflow logging."""

from __future__ import annotations

import contextlib
from typing import Any

import mlflow
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "use_label_encoder": False,
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}


def train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    params: dict | None = None,
    calibrate: bool = True,
    run_name: str = "xgboost-severity",
) -> tuple[Any, dict]:
    """Train an XGBoost multi-class severity classifier and log to the active MLflow run."""
    p = {**DEFAULT_PARAMS, "num_class": num_classes, **(params or {})}

    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.log_params(p)

        clf = XGBClassifier(**p)
        clf.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            verbose=50,
        )

        if calibrate:
            clf = CalibratedClassifierCV(clf, cv="prefit", method="isotonic")
            clf.fit(x_val, y_val)

        metrics = evaluate(clf, x_val, y_val, prefix="val")
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(clf, artifact_path="xgboost-severity")

    return clf, metrics


def evaluate(clf, x: np.ndarray, y: np.ndarray, prefix: str = "test") -> dict:
    y_pred = clf.predict(x)
    y_proba = clf.predict_proba(x)

    macro_f1 = f1_score(y, y_pred, average="macro")
    weighted_f1 = f1_score(y, y_pred, average="weighted")
    kappa = cohen_kappa_score(y, y_pred, weights="quadratic")

    n_classes = y_proba.shape[1]
    roc_auc = float("nan")
    if n_classes == len(np.unique(y)):
        with contextlib.suppress(Exception):
            roc_auc = roc_auc_score(y, y_proba, multi_class="ovr", average="macro")

    report = classification_report(y, y_pred, output_dict=True)
    metrics: dict[str, float] = {
        f"{prefix}_macro_f1": macro_f1,
        f"{prefix}_weighted_f1": weighted_f1,
        f"{prefix}_quadratic_kappa": kappa,
        f"{prefix}_roc_auc_ovr_macro": roc_auc,
    }
    for label, stats in report.items():
        if isinstance(stats, dict):
            for stat_name, val in stats.items():
                metrics[f"{prefix}_cls{label}_{stat_name}"] = val

    return metrics
