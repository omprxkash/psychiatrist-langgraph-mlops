"""
PySpark ETL for the Reddit Mental Health Dataset.

Input format expected
---------------------
The Reddit Mental Health Dataset (https://zenodo.org/records/3941387) ships as per-subreddit
CSV files (one row per post, columns: author, subreddit, body, title, created_utc, ...).
Place them under ``data/raw/reddit_mental_health/`` — one CSV per subreddit is fine.

This job:
1. Reads all CSVs in the input directory.
2. Filters posts to the labelled subreddits we care about
   (r/depression, r/anxiety, r/SuicideWatch, r/mentalhealth, r/Anxiety, r/depression_help, ...).
3. Normalises and cleans text (strip URLs, normalise whitespace, drop deleted/removed posts).
4. Derives a `label` column from subreddit membership (weak supervision).
5. Stratifies a 80/10/10 user-level train/val/test split — *no user leakage*.
6. Writes Parquet partitioned by `split`.

Run:
    python -m spark_jobs.reddit_cohort \\
        --in data/raw/reddit_mental_health \\
        --out data/processed/reddit_cohort.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F  # noqa: N812

LABEL_MAP = {
    "depression": "depression",
    "depression_help": "depression",
    "anxiety": "anxiety",
    "Anxiety": "anxiety",
    "socialanxiety": "anxiety",
    "SuicideWatch": "suicidal_ideation",
    "mentalhealth": "general",
    "MentalHealthSupport": "general",
    "bipolar": "bipolar",
    "BPD": "bpd",
    "ptsd": "ptsd",
}

URL_RE = r"http\S+|www\.\S+"
SUB_RE = r"/?[ru]/\w+"
WHITESPACE_RE = r"\s+"


def load_raw(spark: SparkSession, in_dir: Path):
    return (
        spark.read.option("header", True)
        .option("multiLine", True)
        .option("escape", "\"")
        .csv(str(in_dir / "*.csv"))
    )


def clean(df):
    body_col = F.coalesce(F.col("body"), F.col("selftext"), F.lit(""))
    title_col = F.coalesce(F.col("title"), F.lit(""))
    text = F.concat_ws(" ", title_col, body_col)
    text = F.regexp_replace(text, URL_RE, " ")
    text = F.regexp_replace(text, SUB_RE, " ")
    text = F.regexp_replace(text, WHITESPACE_RE, " ")
    text = F.trim(text)

    return (
        df.withColumn("text", text)
        .filter(F.col("text").isNotNull() & (F.length("text") >= 20))
        .filter(~F.col("text").isin("[deleted]", "[removed]"))
        .filter(F.col("author").isNotNull() & (F.col("author") != "[deleted]"))
        .filter(F.col("subreddit").isin(*LABEL_MAP.keys()))
    )


def label(df):
    mapping_expr = F.create_map(
        *[v for pair in LABEL_MAP.items() for v in (F.lit(pair[0]), F.lit(pair[1]))]
    )
    return df.withColumn("label", mapping_expr[F.col("subreddit")])


def deduplicate(df):
    """Drop duplicate texts (cross-posts and reposts)."""
    return df.dropDuplicates(["author", "text"])


def user_level_split(df, seed: int = 42):
    """Hash users into 80/10/10 buckets so the same author cannot appear in two splits."""
    hashed = F.abs(F.hash(F.col("author"))) % 100
    return df.withColumn(
        "split",
        F.when(hashed < 80, F.lit("train"))
        .when(hashed < 90, F.lit("val"))
        .otherwise(F.lit("test")),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_text_len", type=int, default=20)
    args = parser.parse_args()

    spark = (
        SparkSession.builder.appName("psychiatrist-reddit-cohort")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    raw = load_raw(spark, args.in_dir)
    cleaned = clean(raw)
    labelled = label(cleaned)
    deduped = deduplicate(labelled)
    split = user_level_split(deduped, seed=args.seed)

    output_cols = ["author", "subreddit", "label", "text", "created_utc", "split"]
    available = [c for c in output_cols if c in split.columns]
    final = split.select(*available)

    (
        final.write.mode("overwrite")
        .partitionBy("split")
        .parquet(str(args.out))
    )

    print(f"\nWrote partitioned Parquet to {args.out}\n")
    print("Split sizes:")
    final.groupBy("split").count().orderBy("split").show()
    print("Label distribution by split:")
    final.groupBy("split", "label").count().orderBy("split", F.desc("count")).show(40)

    spark.stop()


if __name__ == "__main__":
    main()
