"""
MentalBERT fine-tuning for two tasks:
  1. Multi-label symptom classification (7 labels)
  2. Binary suicidal-ideation classification (high-recall target)

Both tasks share the same backbone: mental/mental-bert-base-uncased.
They are trained sequentially (not jointly) so each head can have its own
loss function and evaluation metric.

Memory footprint on laptop (no GPU):
  - mental-bert-base-uncased: ~440 MB float32, ~220 MB int8 after quantization.
  - With max_length=256 and batch_size=8, peak RAM is ~4-6 GB.
  - Training on CPU for 3 epochs over ~80k examples takes ~4-8 hours.
  - For faster iteration: use --max-train-samples 5000 during development.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from models.clinical_nlp.dataset import (
    SYMPTOM_LABELS,
    ClinicalTextDataset,
    compute_class_weights_si,
    load_dataset,
)

MODEL_NAME = "mental/mental-bert-base-uncased"


# ── Custom Trainer with weighted loss for SI ───────────────────────────────────

class WeightedBCETrainer(Trainer):
    """Multi-label trainer using per-sample BCE."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction="mean"
        )
        return (loss, outputs) if return_outputs else loss


class HighRecallSITrainer(Trainer):
    """Binary SI trainer with class-weighted cross-entropy (targets high recall)."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss = torch.nn.functional.cross_entropy(logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


# ── Metric functions ──────────────────────────────────────────────────────────

def multilabel_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs > 0.5).astype(int)
    results = {
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }
    for i, label in enumerate(SYMPTOM_LABELS):
        results[f"f1_{label}"] = f1_score(labels[:, i], preds[:, i], zero_division=0)
        results[f"recall_{label}"] = recall_score(labels[:, i], preds[:, i], zero_division=0)
    return results


def si_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    preds = (probs > 0.4).astype(int)  # Lower threshold to bias toward recall.
    return {
        "recall": recall_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "roc_auc": roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0,
    }


# ── Training functions ────────────────────────────────────────────────────────

def train_multilabel(
    data_path: Path,
    output_dir: Path,
    max_train_samples: int | None = None,
    epochs: int = 3,
    batch_size: int = 8,
):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(SYMPTOM_LABELS),
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    )

    train_df = load_dataset(data_path, split="train")
    val_df = load_dataset(data_path, split="val")
    if max_train_samples:
        train_df = train_df.sample(n=min(max_train_samples, len(train_df)), random_state=42)

    train_ds = ClinicalTextDataset(train_df, tokenizer, mode="multilabel")
    val_ds = ClinicalTextDataset(val_df, tokenizer, mode="multilabel")

    training_args = TrainingArguments(
        output_dir=str(output_dir / "multilabel"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=100,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=False,
        no_cuda=not torch.cuda.is_available(),
        report_to="none",
    )

    with mlflow.start_run(run_name="mentalbert-multilabel", nested=True):
        mlflow.log_params(
            {
                "base_model": MODEL_NAME,
                "task": "multilabel_symptoms",
                "epochs": epochs,
                "batch_size": batch_size,
                "n_train": len(train_df),
                "n_val": len(val_df),
            }
        )
        trainer = WeightedBCETrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=multilabel_metrics,
        )
        trainer.train()
        metrics = trainer.evaluate()
        mlflow.log_metrics(metrics)
        model.save_pretrained(output_dir / "multilabel" / "best")
        tokenizer.save_pretrained(output_dir / "multilabel" / "best")
        print(f"\nMulti-label eval metrics: {metrics}")
    return model, tokenizer


def train_si_binary(
    data_path: Path,
    output_dir: Path,
    max_train_samples: int | None = None,
    epochs: int = 3,
    batch_size: int = 8,
):
    """Binary suicidal-ideation classifier. Optimises for recall over precision."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )

    train_df = load_dataset(data_path, split="train")
    val_df = load_dataset(data_path, split="val")
    if max_train_samples:
        train_df = train_df.sample(n=min(max_train_samples, len(train_df)), random_state=42)

    class_weights = compute_class_weights_si(train_df)
    train_ds = ClinicalTextDataset(train_df, tokenizer, mode="binary_si")
    val_ds = ClinicalTextDataset(val_df, tokenizer, mode="binary_si")

    training_args = TrainingArguments(
        output_dir=str(output_dir / "si_binary"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="recall",
        greater_is_better=True,
        logging_steps=100,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=False,
        no_cuda=not torch.cuda.is_available(),
        report_to="none",
    )

    with mlflow.start_run(run_name="mentalbert-si-binary", nested=True):
        pos_weight = class_weights[1].item()
        mlflow.log_params(
            {
                "base_model": MODEL_NAME,
                "task": "binary_suicidal_ideation",
                "epochs": epochs,
                "batch_size": batch_size,
                "positive_class_weight": pos_weight,
                "n_train": len(train_df),
                "n_val": len(val_df),
                "si_positive_in_train": int(train_df["si_label"].sum()),
            }
        )
        trainer = HighRecallSITrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=si_metrics,
            class_weights=class_weights,
        )
        trainer.train()
        metrics = trainer.evaluate()
        mlflow.log_metrics(metrics)
        model.save_pretrained(output_dir / "si_binary" / "best")
        tokenizer.save_pretrained(output_dir / "si_binary" / "best")
        print(f"\nSI binary eval metrics: {metrics}")
        si_recall = metrics.get("eval_recall", 0.0)
        if si_recall < 0.90:
            print(
                f"WARNING: SI recall={si_recall:.3f} is below the 0.90 target. "
                "Consider increasing positive class weight or training longer."
            )
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("models/clinical_nlp/checkpoints"))
    parser.add_argument(
        "--task",
        choices=["multilabel", "si_binary", "all"],
        default="all",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--experiment-name", default="psychiatrist-clinical-nlp")
    args = parser.parse_args()

    mlflow.set_experiment(args.experiment_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=f"mentalbert-{args.task}"):
        if args.task in ("multilabel", "all"):
            train_multilabel(
                args.data_path,
                args.output_dir,
                max_train_samples=args.max_train_samples,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
        if args.task in ("si_binary", "all"):
            train_si_binary(
                args.data_path,
                args.output_dir,
                max_train_samples=args.max_train_samples,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )


if __name__ == "__main__":
    main()
