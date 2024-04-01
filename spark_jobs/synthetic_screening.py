"""
Synthetic PHQ-9 + GAD-7 + demographics generator.

Calibrated against published prevalence distributions:
    - NIMHANS National Mental Health Survey 2015-16 (India): lifetime depression ~5.25%,
      current depressive episode ~2.7%, anxiety disorders ~3.5%.
    - NHANES 2017-2020 (US): PHQ-9 >= 10 prevalence ~8.4%.
    - PHQ-9 / GAD-7 inter-correlation in primary care: r ~ 0.65 (Kroenke et al., 2010).

This generator uses a latent severity variable per subject plus an anxiety-comorbidity
factor, then samples item-level responses from category cutoffs.  This produces a dataset
with realistic item-level correlations rather than independent uniform draws.

Run:
    python -m spark_jobs.synthetic_screening --n 50000 --out data/processed/synthetic_screening.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F  # noqa: N812
from pyspark.sql import types as T  # noqa: N812

PHQ9_ITEMS = [
    "phq1_anhedonia",
    "phq2_low_mood",
    "phq3_sleep",
    "phq4_fatigue",
    "phq5_appetite",
    "phq6_self_worth",
    "phq7_concentration",
    "phq8_psychomotor",
    "phq9_suicidal_ideation",
]

GAD7_ITEMS = [
    "gad1_nervous",
    "gad2_uncontrollable_worry",
    "gad3_excess_worry",
    "gad4_trouble_relaxing",
    "gad5_restless",
    "gad6_irritable",
    "gad7_fearful",
]

# Item difficulty thresholds (latent severity at which response moves from 0->1, 1->2, 2->3).
# Calibrated so that population PHQ-9 totals approximate NHANES/NIMHANS distributions.
PHQ9_THRESHOLDS = np.array(
    [
        [0.6, 1.6, 2.7],  # anhedonia — common
        [0.5, 1.5, 2.6],  # low mood — most common
        [0.7, 1.7, 2.8],  # sleep
        [0.8, 1.8, 2.9],  # fatigue
        [1.0, 2.0, 3.0],  # appetite
        [1.4, 2.3, 3.1],  # self-worth
        [1.5, 2.4, 3.2],  # concentration
        [1.7, 2.6, 3.4],  # psychomotor
        [2.3, 3.1, 3.8],  # suicidal ideation — rarest, highest threshold
    ]
)

GAD7_THRESHOLDS = np.array(
    [
        [-0.3, 0.7, 1.8],  # nervous
        [-0.1, 0.9, 2.0],  # uncontrollable worry
        [0.0, 1.0, 2.1],   # excess worry
        [0.2, 1.2, 2.2],   # trouble relaxing
        [0.3, 1.3, 2.3],   # restless
        [0.4, 1.4, 2.3],   # irritable
        [0.5, 1.5, 2.4],   # fearful something awful
    ]
)


def phq9_severity_band(total: int) -> str:
    """PHQ-9 standard cutoffs (Kroenke et al., 2001)."""
    if total <= 4:
        return "none"
    if total <= 9:
        return "mild"
    if total <= 14:
        return "moderate"
    if total <= 19:
        return "moderately_severe"
    return "severe"


def gad7_severity_band(total: int) -> str:
    """GAD-7 standard cutoffs (Spitzer et al., 2006)."""
    if total <= 4:
        return "minimal"
    if total <= 9:
        return "mild"
    if total <= 14:
        return "moderate"
    return "severe"


def _items(
    latent_value: float,
    thresholds: np.ndarray,
    rng: np.random.Generator,
) -> list[int]:
    noise = rng.standard_normal(len(thresholds)) * 0.6
    shifted = latent_value + noise
    out = np.zeros(len(thresholds), dtype=np.int32)
    for i, row_thresh in enumerate(thresholds):
        out[i] = int((shifted[i] > row_thresh).sum())
    return out.tolist()


def _sample_partition(iterator, depression_thresholds, anxiety_thresholds, seed_base):
    """Generate one subject per input row index using a per-partition RNG."""
    for row in iterator:
        idx = row.idx
        rng = np.random.default_rng(seed_base + idx)

        age = int(rng.integers(18, 80))
        gender = str(rng.choice(["female", "male", "nonbinary"], p=[0.51, 0.48, 0.01]))
        education = str(
            rng.choice(
                ["primary", "secondary", "undergrad", "postgrad"],
                p=[0.15, 0.40, 0.35, 0.10],
            )
        )
        prior_diagnosis = bool(rng.random() < 0.12)
        on_medication = bool(prior_diagnosis and rng.random() < 0.55)

        # Latent severity: standard normal shifted by risk factors.
        latent = float(rng.standard_normal())
        if prior_diagnosis:
            latent += 0.9
        if age < 25:
            latent += 0.15
        if gender == "female":
            latent += 0.10

        # Anxiety latent: correlated with depression latent (r ~ 0.65).
        anxiety_latent = 0.65 * latent + float(rng.standard_normal()) * np.sqrt(1 - 0.65**2)

        phq_items = _items(latent, depression_thresholds, rng)
        gad_items = _items(anxiety_latent, anxiety_thresholds, rng)

        phq_total = int(sum(phq_items))
        gad_total = int(sum(gad_items))

        yield (
            f"SUBJ-{idx:08d}",
            age,
            gender,
            education,
            prior_diagnosis,
            on_medication,
            *phq_items,
            phq_total,
            phq9_severity_band(phq_total),
            *gad_items,
            gad_total,
            gad7_severity_band(gad_total),
            round(latent, 4),
        )


def build_schema() -> T.StructType:
    fields = [
        T.StructField("subject_id", T.StringType(), False),
        T.StructField("age", T.IntegerType(), False),
        T.StructField("gender", T.StringType(), False),
        T.StructField("education", T.StringType(), False),
        T.StructField("prior_diagnosis", T.BooleanType(), False),
        T.StructField("on_medication", T.BooleanType(), False),
    ]
    for item in PHQ9_ITEMS:
        fields.append(T.StructField(item, T.IntegerType(), False))
    fields.append(T.StructField("phq9_total", T.IntegerType(), False))
    fields.append(T.StructField("phq9_band", T.StringType(), False))
    for item in GAD7_ITEMS:
        fields.append(T.StructField(item, T.IntegerType(), False))
    fields.append(T.StructField("gad7_total", T.IntegerType(), False))
    fields.append(T.StructField("gad7_band", T.StringType(), False))
    fields.append(T.StructField("latent_severity", T.DoubleType(), False))
    return T.StructType(fields)


def generate(spark: SparkSession, n: int, seed: int = 42, partitions: int = 8):
    indices = spark.range(0, n, numPartitions=partitions).withColumnRenamed("id", "idx")
    schema = build_schema()
    rows = indices.rdd.mapPartitions(
        lambda it: _sample_partition(it, PHQ9_THRESHOLDS, GAD7_THRESHOLDS, seed)
    )
    return spark.createDataFrame(rows, schema=schema)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50_000, help="Number of synthetic subjects")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--partitions", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True, help="Output Parquet path")
    args = parser.parse_args()

    spark = (
        SparkSession.builder.appName("psychiatrist-synthetic-screening")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", str(args.partitions))
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    df = generate(spark, n=args.n, seed=args.seed, partitions=args.partitions)
    df.write.mode("overwrite").parquet(str(args.out))

    # Quick sanity-check report.
    print(f"\nWrote {args.n:,} rows to {args.out}\n")
    print("PHQ-9 band distribution:")
    df.groupBy("phq9_band").count().orderBy(F.desc("count")).show()
    print("GAD-7 band distribution:")
    df.groupBy("gad7_band").count().orderBy(F.desc("count")).show()
    print(
        "Suicidal-ideation item (phq9_suicidal_ideation > 0) prevalence by phq9 band:"
    )
    df.groupBy("phq9_band").agg(
        F.mean((F.col("phq9_suicidal_ideation") > 0).cast("double")).alias("si_rate"),
        F.count("*").alias("n"),
    ).orderBy("phq9_band").show()

    spark.stop()


if __name__ == "__main__":
    main()
