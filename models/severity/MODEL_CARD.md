# Model Card — Psychiatrist Severity Classifier

> This document follows the Model Cards for Model Reporting framework (Mitchell et al., 2019).

---

## Model overview

**Model name:** `psychiatrist-severity-classifier`
**Version:** See MLflow Model Registry for the current `Production` version.
**Task:** Multi-class classification of PHQ-9 depression severity (none / mild / moderate / moderately severe / severe) from structured screening data (PHQ-9 items, GAD-7 items, demographics).

**Primary model:** XGBoost with isotonic calibration.
**Baseline models:** LightGBM (calibrated), PyTorch tabular MLP.
**Auxiliary task:** Ordinal regression for continuous PHQ-9 total (enables severity-ordered error analysis).

---

## Intended use

**Primary use case:** Educational demonstration of a production-grade ML pipeline in a portfolio project. This model is NOT intended for clinical use.

**Intended users:** Data scientists, ML engineers reviewing portfolio work; interviewers evaluating the candidate's ability to build, evaluate, and deploy an ML model.

**Out-of-scope uses:**
- Clinical screening, diagnosis, or triage of any human patient.
- Any deployment without ethics review, IRB approval, and licensed clinical oversight.
- Use with real patient data without appropriate data-sharing agreements and privacy protections.

---

## Training data

| Source | Type | N | Notes |
|---|---|---|---|
| Synthetic PHQ-9/GAD-7 generator | Latent-variable simulation | 50,000 subjects | Calibrated to NHANES 2017-20 and NIMHANS 2015-16 prevalence distributions |

**Why synthetic data?** Real PHQ-9 datasets with item-level responses are either proprietary (health systems) or require IRB approval. Synthetic data allows full reproducibility and open sharing while demonstrating the complete ML pipeline.

**Known limitation:** A model trained on synthetic data will likely underperform on real populations. Do not deploy without retraining on a real, consented, clinically-validated dataset.

**Split strategy:** 80/10/10 train/val/test split by subject_id hash — no data leakage.

---

## Evaluation

### Headline metrics (val set, synthetic data)

*Populate this table after running `make train`. The values below are illustrative.*

| Model | Macro F1 | Weighted F1 | Quadratic κ | ROC-AUC (OvR) |
|---|---|---|---|---|
| XGBoost (calibrated) | TBD | TBD | TBD | TBD |
| LightGBM (calibrated) | TBD | TBD | TBD | TBD |
| PyTorch MLP | TBD | TBD | TBD | TBD |
| Ordinal Ridge (baseline) | TBD | TBD | TBD | — |

*Artifacts: `confusion_matrix.png`, `calibration_curves.png`, `shap_*.png` — see MLflow experiment `psychiatrist-severity`.*

### Subgroup fairness slices

*Populate this table after running `make train`. Disparity = max(F1) - min(F1) across subgroup values.*

| Dimension | Groups | F1 disparity | Notes |
|---|---|---|---|
| Gender | female / male / nonbinary | TBD | Nonbinary group small (n ≈ 50) — interpret with caution |
| Age band | 18-29 / 30-44 / 45-59 / 60+ | TBD | Older adults may have different symptom presentations |

*Full table: `fairness_slices.csv` in MLflow artifacts.*

**Fairness target:** F1 disparity < 0.05 across gender and age bands. If disparity exceeds this threshold, retrain with stratified sampling or reweighting before promoting to Production.

---

## Limitations and risks

1. **Synthetic training data** — distribution reflects the generator's assumptions, not a real clinical population.
2. **English-only** — all input text in the clinical NLP module (downstream) is English; the tabular model is language-agnostic but was not validated on non-English-speaking populations.
3. **No temporal validation** — the model is not validated against longitudinal data or clinical outcomes.
4. **PHQ-9 self-report is not a diagnosis** — PHQ-9 ≥ 10 is a screening threshold, not a diagnostic criterion. The model's severity output should never be presented as a clinical diagnosis.
5. **Missing values** — the model expects all 9 PHQ-9 items and 7 GAD-7 items to be present. Imputation is not currently implemented.

---

## Quantitative analysis (SHAP)

Top features by mean |SHAP| (populate from `shap_*.png` artifacts after training):

| Rank | Feature | Direction |
|---|---|---|
| 1 | phq9_total | + |
| 2 | phq2_low_mood | + |
| 3 | phq1_anhedonia | + |
| ... | ... | ... |

SHAP interaction values available via `compute_shap()` in `models/severity/evaluation.py`.

---

## Ethical considerations

- This model was built with awareness of the sensitivity of mental health data.
- The training data is public and de-identified (synthetic).
- The model should never be used to make autonomous decisions about a person's care.
- Disparate performance across demographic groups (see fairness slices above) must be disclosed to any stakeholder who reviews or uses the model.

---

## Versioning and retraining

- Model versions are tracked in the MLflow Model Registry under `psychiatrist-severity-classifier`.
- Retraining is triggered automatically by the GitHub Actions drift-detection workflow when Evidently detects significant input or prediction drift (see `monitoring/drift_report.py`).
- Every retraining run updates this model card's metrics table and fairness slice table.
- The suicide-risk regression test suite (`tests/safety/`) must pass before any model is promoted from Staging to Production.

---

## Citation

If you reference this work, please cite:
> Harisankar, "Psychiatrist: Agentic Mental Health Triage and Clinical Decision Support System (Portfolio Project)", 2026. GitHub: [repository URL].
