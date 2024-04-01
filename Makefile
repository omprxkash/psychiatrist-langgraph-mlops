.PHONY: help install data train rag-index test safety lint format clean serve monitor

PYTHON := python
SPARK_SUBMIT := spark-submit

help:
	@echo "Psychiatrist — make targets"
	@echo ""
	@echo "  install      Install package + dev deps + pre-commit hooks"
	@echo "  data         Generate synthetic data + run PySpark ETL"
	@echo "  train        Train severity classifier + MentalBERT fine-tune"
	@echo "  rag-index    Build FAISS index for DSM summaries + PubMed abstracts"
	@echo "  test         Run full pytest suite"
	@echo "  safety       Run safety regression suite only (must pass 100%)"
	@echo "  lint         Run ruff + mypy"
	@echo "  format       Run black + ruff --fix"
	@echo "  serve        Start Streamlit clinical UI (port 8501)"
	@echo "  monitor      Run drift report + safety audit"
	@echo "  clean        Remove caches and build artifacts"

install:
	pip install -e ".[dev]"
	pre-commit install
	pre-commit install --hook-type pre-push

data:
	$(PYTHON) data/generate.py --n 50000 --out data/processed

train:
	$(PYTHON) -m models.train --task severity --model xgboost
	$(PYTHON) -m models.train --task severity --model torch_mlp
	$(PYTHON) -m models.train --task clinical_nlp --model mentalbert

rag-index:
	$(PYTHON) -m rag.ingest_dsm_summaries --out rag/indexes/dsm
	$(PYTHON) -m rag.ingest_pubmed --query "psychiatry[MeSH]" --max 100000 --out rag/indexes/pubmed

test:
	pytest tests/ -v --cov=psychiatrist --cov=agents --cov=models --cov=rag --cov-report=term-missing

safety:
	pytest tests/safety/ -v -m safety

lint:
	ruff check .
	mypy psychiatrist agents models rag serving monitoring mlops spark_jobs

format:
	black .
	ruff check --fix .

serve:
	streamlit run serving/app.py --server.port 8501

monitor:
	$(PYTHON) monitoring/drift_report.py
	$(PYTHON) monitoring/safety_audit.py

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ .ruff_cache/ .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
