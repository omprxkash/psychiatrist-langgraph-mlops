# Safety, Scope, and Responsible Use

## This is NOT a medical device.

Psychiatrist is a **portfolio engineering project** demonstrating an agentic ML/NLP pipeline. It is:

- **Not** FDA / CDSCO / CE approved.
- **Not** a substitute for assessment by a licensed mental health professional.
- **Not** suitable for use with real patient data outside a research environment with appropriate ethics approval.
- **Not** validated for diagnostic, screening, or triage use in any clinical setting.

The outputs of this system — including risk scores, symptom signals, DSM-5 differential considerations, and care-plan suggestions — are **demonstrations of an engineering architecture**, not clinical recommendations. Any human-facing UI built on top of Psychiatrist must surface this disclaimer prominently.

## If you or someone you know is in crisis

This project deals with sensitive content. If you are experiencing a mental health emergency, or are concerned about suicide, self-harm, or someone's immediate safety, please reach a trained human now.

### India

| Service | Number / Channel | Hours |
|---|---|---|
| **iCall** (Tata Institute of Social Sciences) | +91 9152987821 | Mon–Sat, 8 AM–10 PM |
| **Vandrevala Foundation** | 1860-2662-345 / +91 9999666555 | 24/7 |
| **AASRA** | +91 9820466726 | 24/7 |
| **NIMHANS Helpline** | 080-46110007 | 24/7 |
| **iCall Email** | icall@tiss.edu | Within 24 hours |
| **KIRAN (Govt. of India)** | 1800-599-0019 | 24/7, multilingual |

### International

- **International Association for Suicide Prevention** — directory of crisis lines by country: <https://www.iasp.info/resources/Crisis_Centres/>
- **988** (United States) — Suicide & Crisis Lifeline.
- **116 123** (Samaritans, UK & Ireland).

If someone is in immediate danger, contact local emergency services (**112** in India, **911** in US, **999** in UK).

## Data handling

- **Do not** commit any real patient data, identifiable clinical notes, or PHI to this repository.
- All training data used by this project is **public, de-identified, or synthetically generated**:
  - Reddit Mental Health Dataset (public, pseudonymous).
  - Kaggle Suicide & Depression Detection dataset (public, pseudonymous).
  - CLPsych shared-task public subsets.
  - PubMed abstracts (public).
  - Synthetic PHQ-9 / GAD-7 records generated locally.
- DSM-5-TR is copyrighted by the American Psychiatric Association. This repository indexes **paraphrased clinician-style quick-reference summaries**, not the canonical text. Raw DSM-5-TR content is never committed, distributed, or surfaced verbatim in outputs.
- If you fork this project and connect it to any source of real patient data, **stop** — get an IRB/ethics approval, a data-sharing agreement, and a licensed clinician in the loop first.

## Safety-Critic Agent — what it is and is not

The Safety-Critic Agent in `agents/safety_critic.py` is a **defensive engineering pattern**, not a safety guarantee:

- Its deterministic rules are designed to **fail safe**: any positive signal on suicidal ideation forces an `escalate` verdict, regardless of what the upstream agents conclude.
- Its LLM verifier flags hallucinated symptoms, missing citations, and inappropriately confident phrasing.
- It is **not** a substitute for a human reviewer. It does not eliminate false negatives. It does not guarantee that escalations are followed up. It does not absolve a deployer of clinical, legal, or ethical responsibility for the system's behavior.

If you change the deterministic rules in `safety_critic.py`, you must:
1. Update the suicide-risk regression test set under `tests/safety/`.
2. Document the change in `SAFETY.md` (this file).
3. Get review from at least one other person.

The regression test set targets **100% recall** on a curated suite of overt and subtle suicidal-ideation examples. Any PR that lowers recall on this suite must be rejected.

## Bias, fairness, and limitations

- Training data is heavily skewed toward English-speaking Reddit users; the model will not generalize to other languages, cultures, or clinical settings without retraining.
- Self-report data is a poor proxy for clinical assessment; the system will both over- and under-detect compared to a trained clinician.
- Subgroup performance (age, gender) is reported in `models/severity/MODEL_CARD.md`. Read it before drawing conclusions from any single prediction.

## Contact

This is a personal portfolio project. For questions, open a GitHub issue. Do **not** send unsolicited clinical data.
