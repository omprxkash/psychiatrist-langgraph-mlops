"""
Unified training CLI.

Usage examples:
    python -m models.train --task severity --model xgboost
    python -m models.train --task severity --model lightgbm
    python -m models.train --task severity --model torch_mlp
    python -m models.train --task severity --model all
    python -m models.train --task clinical_nlp --nlp-model multilabel
    python -m models.train --task clinical_nlp --nlp-model si_binary
    python -m models.train --task clinical_nlp --nlp-model all --max-train-samples 5000
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import mlflow
import pandas as pd

from models.severity.features import SEVERITY_BANDS, prepare


def _train_severity(args: argparse.Namespace):
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data not found at {data_path}. Run `make data` first to generate the synthetic dataset."
        )

    print(f"Loading data from {data_path} ...")
    df = pd.read_parquet(data_path)
    prepared = prepare(df, target_col="phq9_band", split_col="split")

    splits = prepared["splits"]
    preprocessor = prepared["preprocessor"]
    le = prepared["label_encoder"]
    feature_names = prepared["feature_names"]
    num_classes = len(SEVERITY_BANDS)
    class_labels = list(le.classes_)

    x_tr, y_tr = splits["train"]["X"], splits["train"]["y"]
    y_ord_tr = splits["train"]["y_ordinal"]
    x_val, y_val = splits["val"]["X"], splits["val"]["y"]
    y_ord_val = splits["val"]["y_ordinal"]
    x_te, y_te = splits["test"]["X"], splits["test"]["y"]
    df_val_meta = splits["val"]["df"]

    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=f"severity-{args.model}"):
        mlflow.log_param("model", args.model)
        mlflow.log_param("data_path", str(data_path))
        mlflow.log_param("n_train", len(x_tr))
        mlflow.log_param("n_val", len(x_val))
        mlflow.log_param("n_test", len(x_te))
        mlflow.log_param("num_features", x_tr.shape[1])

        # Persist preprocessing artifacts
        artifact_dir = Path("mlartifacts") / "preprocessor"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        with open(artifact_dir / "preprocessor.pkl", "wb") as f:
            pickle.dump(preprocessor, f)
        with open(artifact_dir / "label_encoder.pkl", "wb") as f:
            pickle.dump(le, f)
        with open(artifact_dir / "feature_names.json", "w") as f:
            json.dump(feature_names, f)
        mlflow.log_artifacts(str(artifact_dir), artifact_path="preprocessor")

        models_trained = []

        if args.model in ("xgboost", "all"):
            from models.severity.xgboost_model import (
                evaluate as xgb_eval,
            )
            from models.severity.xgboost_model import (
                train as xgb_train,
            )
            print("\n── XGBoost ──")
            xgb_clf, xgb_val_metrics = xgb_train(
                x_tr, y_tr, x_val, y_val,
                num_classes=num_classes,
                run_name="xgboost-severity",
            )
            xgb_test_metrics = xgb_eval(xgb_clf, x_te, y_te, prefix="test")
            mlflow.log_metrics(xgb_test_metrics)
            models_trained.append(("xgboost", xgb_clf))
            print(f"XGBoost val macro F1: {xgb_val_metrics.get('val_macro_f1', 'n/a'):.4f}")

        if args.model in ("lightgbm", "all"):
            from lightgbm import LGBMClassifier
            from sklearn.calibration import CalibratedClassifierCV

            from models.severity.xgboost_model import evaluate as lgb_eval
            print("\n── LightGBM ──")
            with mlflow.start_run(run_name="lightgbm-severity", nested=True):
                lgb_params = dict(
                    n_estimators=400, max_depth=6, learning_rate=0.05,
                    num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                    random_state=42, n_jobs=-1, verbose=-1,
                )
                mlflow.log_params(lgb_params)
                lgb_clf = LGBMClassifier(**lgb_params)
                lgb_clf.fit(x_tr, y_tr, eval_set=[(x_val, y_val)])
                lgb_clf = CalibratedClassifierCV(lgb_clf, cv="prefit", method="isotonic")
                lgb_clf.fit(x_val, y_val)
                lgb_val_m = lgb_eval(lgb_clf, x_val, y_val, prefix="val")
                mlflow.log_metrics(lgb_val_m)
                mlflow.sklearn.log_model(lgb_clf, artifact_path="lightgbm-severity")
            models_trained.append(("lightgbm", lgb_clf))
            print(f"LightGBM val macro F1: {lgb_val_m.get('val_macro_f1', 'n/a'):.4f}")

        if args.model in ("torch_mlp", "all"):
            from models.severity.torch_mlp import (
                MLPConfig,
            )
            from models.severity.torch_mlp import (
                train as mlp_train,
            )
            print("\n── PyTorch MLP ──")
            cfg = MLPConfig(
                input_dim=x_tr.shape[1],
                num_classes=num_classes,
                hidden_dims=[256, 128, 64],
                dropout=0.3,
                lr=1e-3,
                epochs=60,
                batch_size=512,
            )
            mlp_model, mlp_metrics = mlp_train(
                x_tr, y_tr, y_ord_tr, x_val, y_val, y_ord_val, cfg,
                run_name="torch-mlp-severity",
            )
            models_trained.append(("torch_mlp", mlp_model))
            print(f"MLP val macro F1: {mlp_metrics.get('val_macro_f1', 'n/a'):.4f}")

        # Evaluation for the best available model (first in list = xgboost preferred)
        if models_trained:
            best_name, best_clf = models_trained[0]
            print(f"\n── Full evaluation on val set (model: {best_name}) ──")
            from models.severity.evaluation import (
                compute_fairness_slices,
                compute_shap,
                log_calibration_curves,
                log_confusion_matrix,
                log_fairness_report,
                log_shap_plots,
            )

            y_pred_val = best_clf.predict(x_val)
            log_confusion_matrix(y_val, y_pred_val, class_labels)
            log_calibration_curves(best_clf, x_val, y_val, class_labels)

            print("Computing SHAP values (may take a moment) ...")
            try:
                shap_vals = compute_shap(best_clf, x_val, feature_names)
                log_shap_plots(shap_vals, feature_names, class_labels)
            except Exception as e:
                print(f"SHAP skipped: {e}")

            fairness_df = compute_fairness_slices(best_clf, x_val, y_val, df_val_meta)
            log_fairness_report(fairness_df)
            print("\nFairness slices:")
            print(fairness_df.to_string(index=False))

            # Promote best model to MLflow registry
            if args.register:
                client = mlflow.tracking.MlflowClient()
                run_id = mlflow.active_run().info.run_id
                model_uri = f"runs:/{run_id}/{best_name}-severity"
                mv = mlflow.register_model(model_uri, name="psychiatrist-severity-classifier")
                client.transition_model_version_stage(
                    name="psychiatrist-severity-classifier",
                    version=mv.version,
                    stage="Staging",
                )
                print(f"\nRegistered {best_name} as model version {mv.version} (Staging)")


def _train_clinical_nlp(args: argparse.Namespace):
    from models.clinical_nlp.finetune import train_multilabel, train_si_binary

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data not found at {data_path}. Run `make data` first."
        )

    mlflow.set_experiment(args.experiment_name)
    output_dir = Path("models/clinical_nlp/checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=f"clinical-nlp-{args.nlp_model}"):
        if args.nlp_model in ("multilabel", "all"):
            train_multilabel(
                data_path,
                output_dir,
                max_train_samples=args.max_train_samples,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
        if args.nlp_model in ("si_binary", "all"):
            train_si_binary(
                data_path,
                output_dir,
                max_train_samples=args.max_train_samples,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )

        if args.quantize:
            from models.clinical_nlp.quantize import quantize_model
            for task_dir in ["multilabel", "si_binary"]:
                src = output_dir / task_dir / "best"
                if src.exists():
                    quantize_model(src, output_dir / task_dir / "int8")


def main():
    parser = argparse.ArgumentParser(description="Psychiatrist model training")
    sub = parser.add_subparsers(dest="task", required=True)

    sev = sub.add_parser("severity", help="Train PHQ-9 severity classifier")
    sev.add_argument(
        "--model",
        choices=["xgboost", "lightgbm", "torch_mlp", "all"],
        default="all",
    )
    sev.add_argument(
        "--data-path",
        default="data/processed/synthetic_screening.parquet",
    )
    sev.add_argument("--experiment-name", default="psychiatrist-severity")
    sev.add_argument("--register", action="store_true")

    nlp = sub.add_parser("clinical_nlp", help="Fine-tune MentalBERT for symptom NLP")
    nlp.add_argument(
        "--nlp-model",
        choices=["multilabel", "si_binary", "all"],
        default="all",
    )
    nlp.add_argument(
        "--data-path",
        default="data/processed/reddit_cohort.parquet",
    )
    nlp.add_argument("--max-train-samples", type=int, default=None,
                     help="Limit training samples for fast dev iteration")
    nlp.add_argument("--epochs", type=int, default=3)
    nlp.add_argument("--batch-size", type=int, default=8)
    nlp.add_argument("--quantize", action="store_true",
                     help="Quantize to int8 after training")
    nlp.add_argument("--experiment-name", default="psychiatrist-clinical-nlp")

    args = parser.parse_args()

    if args.task == "severity":
        _train_severity(args)
    elif args.task == "clinical_nlp":
        _train_clinical_nlp(args)
    else:
        parser.error(f"Unknown task: {args.task}")


if __name__ == "__main__":
    main()
