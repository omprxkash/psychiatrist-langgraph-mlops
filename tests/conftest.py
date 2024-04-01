"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def spark():
    """Provide a local Spark session for tests that need one.  Lazily imports pyspark."""
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.appName("psychiatrist-tests")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.driver.memory", "1g")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()
