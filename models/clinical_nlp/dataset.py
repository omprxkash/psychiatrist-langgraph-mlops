"""
Dataset loader and label encoder for the clinical NLP models.

Reads the Reddit Mental Health Parquet (output of spark_jobs/reddit_cohort.py)
and maps weak subreddit labels to clinical symptom labels suitable for
multi-label classification.

Label scheme
------------
We train two heads:

1. **Symptom multi-label** — 7 binary labels derived from subreddit + heuristic
   keyword signals:
       depression, anxiety, suicidal_ideation, substance_use,
       insomnia, anhedonia, hopelessness

2. **Suicidal ideation binary** — HIGH RECALL target.
   Positive = r/SuicideWatch OR keyword-triggered from the symptom signal.
   The binary head is trained separately with class-weighted loss so false
   negatives are heavily penalised.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

SYMPTOM_LABELS = [
    "depression",
    "anxiety",
    "suicidal_ideation",
    "substance_use",
    "insomnia",
    "anhedonia",
    "hopelessness",
]

# Subreddit → which symptom labels to activate (weak supervision)
SUBREDDIT_LABEL_MAP: dict[str, list[str]] = {
    "depression": ["depression", "anhedonia", "hopelessness"],
    "depression_help": ["depression", "hopelessness"],
    "anxiety": ["anxiety"],
    "Anxiety": ["anxiety"],
    "socialanxiety": ["anxiety"],
    "SuicideWatch": ["suicidal_ideation", "hopelessness"],
    "mentalhealth": ["depression"],
    "MentalHealthSupport": ["depression"],
    "bipolar": ["depression", "anhedonia"],
    "BPD": ["depression", "anxiety"],
    "ptsd": ["anxiety"],
}

# Keyword heuristics for augmenting weak labels
KEYWORD_SIGNALS: dict[str, list[str]] = {
    "suicidal_ideation": [
        "kill myself", "end my life", "want to die", "suicide", "suicidal",
        "take my own life", "not want to live", "rather be dead",
    ],
    "insomnia": [
        "can't sleep", "cannot sleep", "insomnia", "up all night", "no sleep",
        "awake at", "3am", "4am",
    ],
    "substance_use": [
        "drinking", "drunk", "alcohol", "weed", "marijuana", "drugs",
        "high to cope", "self-medicate",
    ],
    "anhedonia": [
        "nothing feels good", "can't enjoy", "lost interest", "no pleasure",
        "things I used to love",
    ],
    "hopelessness": [
        "no point", "pointless", "hopeless", "never get better", "no future",
        "give up",
    ],
}

MAX_LENGTH = 256


def _text_to_labels(text: str, subreddit: str) -> list[int]:
    base = set(SUBREDDIT_LABEL_MAP.get(subreddit, []))
    text_lower = text.lower()
    for symptom, kws in KEYWORD_SIGNALS.items():
        if any(kw in text_lower for kw in kws):
            base.add(symptom)
    return [int(s in base) for s in SYMPTOM_LABELS]


def load_dataset(parquet_path: str | Path, split: str = "train") -> pd.DataFrame:
    df = pd.read_parquet(str(parquet_path), filters=[("split", "=", split)])
    df = df.dropna(subset=["text", "subreddit"]).copy()
    df["labels"] = df.apply(lambda r: _text_to_labels(r["text"], r["subreddit"]), axis=1)
    df["si_label"] = df["labels"].apply(lambda row: row[SYMPTOM_LABELS.index("suicidal_ideation")])
    return df


class ClinicalTextDataset(Dataset):
    """Wraps a DataFrame of text + multi-label targets for the HuggingFace Trainer."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int = MAX_LENGTH,
        mode: str = "multilabel",  # "multilabel" | "binary_si"
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            row["text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.mode == "multilabel":
            item["labels"] = torch.tensor(row["labels"], dtype=torch.float32)
        else:
            item["labels"] = torch.tensor(row["si_label"], dtype=torch.long)
        return item


def compute_class_weights_si(df: pd.DataFrame) -> torch.Tensor:
    """High-recall objective: weight positives heavily for suicidal-ideation."""
    n_pos = df["si_label"].sum()
    n_neg = len(df) - n_pos
    if n_pos == 0:
        return torch.tensor([1.0, 1.0])
    ratio = n_neg / n_pos
    # Weight positives even more than the natural ratio to target high recall.
    positive_weight = min(ratio * 2.0, 20.0)
    return torch.tensor([1.0, positive_weight])
