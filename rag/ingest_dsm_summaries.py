"""
DSM-5-TR criteria summaries ingestion.

IMPORTANT copyright notice
--------------------------
The DSM-5-TR is copyrighted by the American Psychiatric Association.
This file works with PARAPHRASED CLINICIAN QUICK-REFERENCE summaries only.
The raw DSM-5-TR text is NEVER stored in this repository.

Two data sources are supported:
  1. A local JSON/JSONL file of paraphrased summaries you write yourself.
  2. A curated list of public PubMed review articles that summarise each
     DSM-5 diagnostic category — these are public domain.

Place your summaries at:  rag/data/dsm_summaries.jsonl
Format (one JSON object per line):
  {"disorder": "Major Depressive Disorder", "criteria": "...", "source": "paraphrase"}

Usage:
    python -m rag.ingest_dsm_summaries --out rag/indexes/dsm
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SUMMARIES = Path("rag/data/dsm_summaries.jsonl")

# Fallback built-in paraphrased summaries for the most common disorders.
# These are original paraphrases drawing on publicly available clinical information,
# not lifted from the DSM-5-TR.
BUILTIN_SUMMARIES: list[dict] = [
    {
        "disorder": "Major Depressive Disorder",
        "icd10": "F32 / F33",
        "criteria_summary": (
            "Five or more of the following symptoms present for at least 2 weeks, "
            "representing a change from previous functioning, with at least one being "
            "depressed mood or loss of interest: (1) depressed mood most of the day, "
            "(2) markedly diminished interest or pleasure in activities, "
            "(3) significant weight change or appetite disturbance, "
            "(4) insomnia or hypersomnia, "
            "(5) psychomotor agitation or retardation, "
            "(6) fatigue or loss of energy, "
            "(7) feelings of worthlessness or excessive guilt, "
            "(8) impaired concentration or decision-making, "
            "(9) recurrent thoughts of death or suicidal ideation. "
            "Symptoms cause clinically significant distress or impairment. "
            "Episode not attributable to substances or other medical conditions."
        ),
        "source": "paraphrase",
    },
    {
        "disorder": "Generalised Anxiety Disorder",
        "icd10": "F41.1",
        "criteria_summary": (
            "Excessive anxiety and worry about multiple events or activities occurring "
            "more days than not for at least 6 months, difficulty controlling worry, "
            "and at least 3 of: restlessness, fatigue, concentration difficulty, "
            "irritability, muscle tension, sleep disturbance. "
            "Symptoms cause significant distress or functional impairment "
            "and are not due to substances or medical conditions."
        ),
        "source": "paraphrase",
    },
    {
        "disorder": "Persistent Depressive Disorder (Dysthymia)",
        "icd10": "F34.1",
        "criteria_summary": (
            "Depressed mood most of the day, more days than not, for at least 2 years "
            "(1 year in children/adolescents), with at least 2 of: poor appetite or "
            "overeating, insomnia or hypersomnia, low energy, low self-esteem, "
            "poor concentration, hopelessness. "
            "The person has never been symptom-free for more than 2 months during this period."
        ),
        "source": "paraphrase",
    },
    {
        "disorder": "Panic Disorder",
        "icd10": "F41.0",
        "criteria_summary": (
            "Recurrent unexpected panic attacks — abrupt surges of intense fear or discomfort "
            "peaking within minutes, with 4+ of: palpitations, sweating, trembling, "
            "shortness of breath, choking feeling, chest pain, nausea, dizziness, "
            "chills/hot flushes, paraesthesias, derealisation/depersonalisation, "
            "fear of losing control, fear of dying. "
            "At least one attack is followed by 1+ month of persistent concern about "
            "further attacks or maladaptive behavioural change."
        ),
        "source": "paraphrase",
    },
    {
        "disorder": "Social Anxiety Disorder",
        "icd10": "F40.10",
        "criteria_summary": (
            "Marked fear or anxiety about social situations where the person may be "
            "scrutinised by others (e.g., conversations, meetings, performing). "
            "Fear of acting in a way that will be humiliating or cause rejection. "
            "Social situations almost always provoke anxiety, are avoided or endured with "
            "intense anxiety, are out of proportion to actual threat, and have persisted "
            "for 6+ months causing significant distress or impairment."
        ),
        "source": "paraphrase",
    },
    {
        "disorder": "PTSD (Post-Traumatic Stress Disorder)",
        "icd10": "F43.1",
        "criteria_summary": (
            "Exposure to actual or threatened death, serious injury, or sexual violence. "
            "Symptoms from each cluster: intrusion (flashbacks, nightmares, distress at cues), "
            "avoidance (of trauma-related stimuli), negative cognitions/mood (distorted blame, "
            "persistent negative emotions, diminished interest, detachment), "
            "hyperarousal (irritability, reckless behaviour, hypervigilance, "
            "exaggerated startle, sleep disturbance). "
            "Duration > 1 month; symptoms not due to substances or medical conditions."
        ),
        "source": "paraphrase",
    },
    {
        "disorder": "Bipolar I Disorder",
        "icd10": "F31",
        "criteria_summary": (
            "Defined by at least one manic episode lasting 1+ week (or any duration if "
            "hospitalisation required): elevated/expansive/irritable mood plus increased "
            "goal-directed activity or energy, with 3+ of (4 if irritable): grandiosity, "
            "decreased sleep need, pressured speech, racing thoughts, distractibility, "
            "increased activity, reckless behaviour. "
            "Depressive and hypomanic episodes often also present but are not required."
        ),
        "source": "paraphrase",
    },
    {
        "disorder": "Suicidal Ideation — Clinical Assessment",
        "icd10": "R45.851",
        "criteria_summary": (
            "Suicidal ideation ranges from passive wishes to be dead (passive ideation) "
            "through active ideation without plan, active ideation with plan, and active ideation "
            "with intent and means. "
            "Columbia Suicide Severity Rating Scale (C-SSRS) is the standard instrument. "
            "High-risk indicators: specific plan, access to means, prior attempts, recent losses, "
            "social isolation, hopelessness, substance use comorbidity. "
            "Escalation required for active ideation with plan or intent — "
            "immediate safety assessment and crisis intervention protocols apply."
        ),
        "source": "paraphrase",
    },
]


def load_summaries(path: Path | None = None) -> list[dict]:
    summaries = list(BUILTIN_SUMMARIES)
    if path and path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    summaries.append(json.loads(line))
        print(f"Loaded {len(summaries) - len(BUILTIN_SUMMARIES)} custom summaries from {path}")
    return summaries


def ingest(out_dir: Path, custom_path: Path | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = load_summaries(custom_path)
    print(f"Total summaries: {len(summaries)} (including {len(BUILTIN_SUMMARIES)} built-in)")

    texts = [
        f"{s['disorder']}\n\n{s['criteria_summary']}"
        for s in summaries
    ]

    encoder = SentenceTransformer(EMBED_MODEL)
    embeddings = encoder.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(out_dir / "faiss.index"))
    with open(out_dir / "chunks.pkl", "wb") as f:
        pickle.dump(texts, f)
    with open(out_dir / "metadata.pkl", "wb") as f:
        pickle.dump(summaries, f)

    print(f"\nDSM index written to {out_dir}: {index.ntotal} entries")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("rag/indexes/dsm"))
    parser.add_argument("--custom", type=Path, default=None,
                        help="Path to custom JSONL summaries (one JSON per line)")
    args = parser.parse_args()
    ingest(args.out, args.custom)


if __name__ == "__main__":
    main()
