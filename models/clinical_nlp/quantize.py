"""
int8 dynamic quantization of MentalBERT for laptop inference.

PyTorch dynamic quantization replaces Linear layers with int8 equivalents at
runtime.  No calibration data needed.  Reduces model size from ~440 MB to
~220 MB and speeds up CPU inference ~1.5-2x.

Usage:
    python -m models.clinical_nlp.quantize \\
        --model-dir models/clinical_nlp/checkpoints/multilabel/best \\
        --out models/clinical_nlp/checkpoints/multilabel/int8
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def quantize_model(model_dir: Path, out_dir: Path) -> None:
    print(f"Loading model from {model_dir} ...")
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    float_size = _model_size_mb(model)
    print(f"Float32 model size: {float_size:.1f} MB")

    quantized = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={torch.nn.Linear},
        dtype=torch.qint8,
    )

    int8_size = _model_size_mb(quantized)
    print(f"int8 model size:    {int8_size:.1f} MB  ({100*(1-int8_size/float_size):.0f}% reduction)")

    # Sanity-check: run a dummy forward pass.
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    dummy = tokenizer("This is a test.", return_tensors="pt", padding="max_length", max_length=64, truncation=True)
    with torch.no_grad():
        out = quantized(**dummy)
    assert out.logits.shape[1] == model.config.num_labels, "Logit shape mismatch after quantization"
    print("Sanity check passed.")

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(quantized.state_dict(), out_dir / "int8_state_dict.pt")
    tokenizer.save_pretrained(str(out_dir))
    model.config.save_pretrained(str(out_dir))

    (out_dir / "quantization_info.txt").write_text(
        f"base_model_dir: {model_dir}\n"
        f"float32_size_mb: {float_size:.1f}\n"
        f"int8_size_mb: {int8_size:.1f}\n"
        f"method: torch.quantization.quantize_dynamic (qint8, Linear layers)\n"
    )
    print(f"Saved quantized model to {out_dir}")


def _model_size_mb(model: torch.nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1024 / 1024


def load_quantized(model_dir: Path) -> tuple:
    """Load a quantized model from disk for inference."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    quantize_model(args.model_dir, args.out)


if __name__ == "__main__":
    main()
