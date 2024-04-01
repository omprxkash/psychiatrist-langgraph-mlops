"""
Inference module for clinical NLP models.

Provides a unified ClinicalNLPPredictor that loads both:
  - The multi-label symptom classifier
  - The binary suicidal-ideation classifier

and exposes a single predict(text) API consumed by the agents layer.

Example:
    from models.clinical_nlp.predict import ClinicalNLPPredictor, ClinicalNLPResult
    predictor = ClinicalNLPPredictor.from_checkpoints("models/clinical_nlp/checkpoints")
    result = predictor.predict("I have not been sleeping and feel completely hopeless.")
    print(result.suicidal_ideation_prob, result.symptoms)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from models.clinical_nlp.dataset import SYMPTOM_LABELS

SI_THRESHOLD = 0.40  # Lower than 0.5 — biased toward recall for safety.
SYMPTOM_THRESHOLD = 0.50
MAX_LENGTH = 256


@dataclass
class ClinicalNLPResult:
    text: str
    symptoms: dict[str, float] = field(default_factory=dict)
    suicidal_ideation_prob: float = 0.0
    suicidal_ideation_positive: bool = False
    active_symptoms: list[str] = field(default_factory=list)
    raw_symptom_logits: list[float] = field(default_factory=list)


class ClinicalNLPPredictor:
    def __init__(
        self,
        multilabel_model: AutoModelForSequenceClassification,
        multilabel_tokenizer: AutoTokenizer,
        si_model: AutoModelForSequenceClassification,
        si_tokenizer: AutoTokenizer,
    ):
        self._ml_model = multilabel_model.eval()
        self._ml_tok = multilabel_tokenizer
        self._si_model = si_model.eval()
        self._si_tok = si_tokenizer

    @classmethod
    def from_checkpoints(
        cls,
        checkpoint_root: str | Path,
        quantized: bool = True,
    ) -> ClinicalNLPPredictor:
        root = Path(checkpoint_root)
        ml_dir = root / "multilabel" / ("int8" if quantized else "best")
        si_dir = root / "si_binary" / ("int8" if quantized else "best")

        if quantized and (ml_dir / "int8_state_dict.pt").exists():
            ml_model, ml_tok = _load_quantized(ml_dir)
        else:
            ml_model = AutoModelForSequenceClassification.from_pretrained(str(ml_dir))
            ml_tok = AutoTokenizer.from_pretrained(str(ml_dir))

        if quantized and (si_dir / "int8_state_dict.pt").exists():
            si_model, si_tok = _load_quantized(si_dir)
        else:
            si_model = AutoModelForSequenceClassification.from_pretrained(str(si_dir))
            si_tok = AutoTokenizer.from_pretrained(str(si_dir))

        return cls(ml_model, ml_tok, si_model, si_tok)

    def predict(self, text: str) -> ClinicalNLPResult:
        symptom_probs = self._predict_multilabel(text)
        si_prob = self._predict_si(text)

        active_symptoms = [
            label for label, prob in zip(SYMPTOM_LABELS, symptom_probs, strict=False)
            if prob >= SYMPTOM_THRESHOLD
        ]

        return ClinicalNLPResult(
            text=text,
            symptoms=dict(zip(SYMPTOM_LABELS, symptom_probs.tolist(), strict=False)),
            suicidal_ideation_prob=float(si_prob),
            suicidal_ideation_positive=si_prob >= SI_THRESHOLD,
            active_symptoms=active_symptoms,
            raw_symptom_logits=symptom_probs.tolist(),
        )

    def predict_batch(self, texts: list[str]) -> list[ClinicalNLPResult]:
        return [self.predict(t) for t in texts]

    def _predict_multilabel(self, text: str) -> np.ndarray:
        enc = self._ml_tok(
            text,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self._ml_model(**enc).logits
        return torch.sigmoid(logits).squeeze(0).numpy()

    def _predict_si(self, text: str) -> float:
        enc = self._si_tok(
            text,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self._si_model(**enc).logits
        return torch.softmax(logits, dim=-1)[0, 1].item()


def _load_quantized(model_dir: Path):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_config(config)
    model.eval()
    quantized = torch.quantization.quantize_dynamic(
        model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
    )
    state = torch.load(model_dir / "int8_state_dict.pt", map_location="cpu")
    quantized.load_state_dict(state)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    return quantized, tokenizer
