"""
00_smoke_test.py

Verify the environment works end-to-end before doing anything else:
  - SpeechBrain imports
  - Model downloads from HuggingFace
  - We can extract LOGITS (not just probabilities) from one audio clip
  - Logit shape matches expected (1, 16)

"""

from __future__ import annotations

import sys
from pathlib import Path

# Add repo root to path so `from src import ...` works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.constants import (
    MODEL_SOURCE,
    MODEL_SAVEDIR,
    CV_LABELS,
    TARGET_SAMPLE_RATE,
)


def main():
    print(f"Loading model: {MODEL_SOURCE}")

    # SpeechBrain >=1.0 import path
    try:
        from speechbrain.inference.classifiers import EncoderClassifier
    except ImportError:
        # Fallback to old import path (speechbrain <1.0)
        from speechbrain.pretrained import EncoderClassifier

    classifier = EncoderClassifier.from_hparams(
        source=MODEL_SOURCE,
        savedir=MODEL_SAVEDIR,
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    )
    print(f"Model loaded on {next(classifier.mods.parameters()).device}")

    # Verify the label encoder
    label_file = Path(MODEL_SAVEDIR) / "label_encoder.txt"
    if label_file.exists():
        print(f"\nlabel_encoder.txt contents:\n{label_file.read_text()[:500]}")
        print(f"\nCV_LABELS in constants.py: {CV_LABELS}")
        print("\n*** Verify the labels in the file match CV_LABELS in src/constants.py ***")
        print("*** If they don't match, edit constants.py to match the file. ***")

    # Generate one second of silence as a test clip (just to exercise the API)
    n_samples = TARGET_SAMPLE_RATE  # 1 second
    fake_audio = torch.zeros(1, n_samples)

    # classify_batch returns (out_prob, score, index, text_lab)
    # out_prob is what speechbrain's EncoderClassifier returns; we need to
    # check whether this is logits or post-softmax. SpeechBrain typically
    # returns LOG-PROBABILITIES (log_softmax of logits), which is fine —
    # we can convert back to logits with np.log(softmax(...)) if needed.
    out_prob, score, index, text_lab = classifier.classify_batch(fake_audio)
    print(f"\nclassify_batch output shapes:")
    print(f"  out_prob: {tuple(out_prob.shape)}  dtype={out_prob.dtype}")
    print(f"  score:    {tuple(score.shape)}")
    print(f"  index:    {tuple(index.shape)}")
    print(f"  text_lab: {text_lab}")

    print(f"\nout_prob first row (sum should be ~1.0 if probs, or ~0 if log-probs):")
    row = out_prob[0].cpu().numpy()
    print(f"  sum  = {row.sum():.4f}")
    print(f"  exp(row).sum = {np.exp(row).sum():.4f}")
    print(f"  max  = {row.max():.4f}")
    print(f"  min  = {row.min():.4f}")
    print(f"  values: {row}")

    # Heuristic detection
    if abs(row.sum() - 1.0) < 0.01:
        print("\n=> out_prob is PROBABILITIES (sum to 1).")
        print("   To get logits, take log(out_prob).")
    elif abs(np.exp(row).sum() - 1.0) < 0.01:
        print("\n=> out_prob is LOG-PROBABILITIES (exp sums to 1).")
        print("   These are equivalent to logits up to a global shift; use directly")
        print("   for temperature scaling — softmax(logits/T) == softmax((logits+c)/T).")
    else:
        print("\n=> out_prob is something else (raw logits or unnormalized). Inspect.")

    print("\n[OK] Smoke test passed.")


if __name__ == "__main__":
    main()
