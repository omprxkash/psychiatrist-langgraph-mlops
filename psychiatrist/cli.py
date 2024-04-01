"""
Psychiatrist CLI entry point.

Usage:
    psychiatrist serve       # Start the Streamlit UI on :8501
    psychiatrist api         # Start the FastAPI service on :8000
    psychiatrist safety      # Run safety regression suite
    psychiatrist data        # Generate synthetic training data
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer(
    name="psychiatrist",
    help="Agentic mental-health triage — portfolio project.",
    add_completion=False,
)

ROOT = Path(__file__).parent.parent


@app.command()
def serve(port: int = typer.Option(8501, help="Streamlit port")):
    """Start the Streamlit clinical UI."""
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run",
         str(ROOT / "serving" / "app.py"),
         "--server.port", str(port),
         "--server.headless", "true"],
        check=True,
    )


@app.command()
def api(port: int = typer.Option(8000, help="FastAPI port")):
    """Start the FastAPI inference service."""
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "serving.api:app",
         "--reload", "--port", str(port)],
        cwd=str(ROOT),
        check=True,
    )


@app.command()
def safety():
    """Run the safety regression suite (must be 100% recall)."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/safety/", "-v", "-m", "safety"],
        cwd=str(ROOT),
    )
    raise SystemExit(result.returncode)


@app.command()
def data(
    n: int = typer.Option(50_000, help="Number of synthetic records to generate"),
    out: str = typer.Option("data/processed", help="Output directory"),
):
    """Generate synthetic PHQ-9 / GAD-7 training data."""
    subprocess.run(
        [sys.executable, str(ROOT / "data" / "generate.py"),
         "--n", str(n), "--out", out],
        cwd=str(ROOT),
        check=True,
    )
