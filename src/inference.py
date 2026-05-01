"""
Model loading and logit extraction.

The CommonAccent model returns log-probabilities by default. For temperature
scaling we want pre-softmax logits, but log-probabilities are equivalent up
to a global additive constant (softmax(x/T) is invariant to additive
shifts), so we can use them directly.

This module exposes:
    load_classifier()     -> EncoderClassifier
    extract_logits(audio) -> (K,) numpy array

Internally we run inference one clip at a time. EdAcc is small (~40 hours);
batching is a stretch optimization.
"""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path

from src.constants import MODEL_SOURCE, MODEL_SAVEDIR


_classifier_cache = None


def load_classifier(device: str | None = None):
    """
    Load the SpeechBrain EncoderClassifier (cached for repeated calls).
    """
    global _classifier_cache
    if _classifier_cache is not None:
        return _classifier_cache

    try:
        from speechbrain.inference.classifiers import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    classifier = EncoderClassifier.from_hparams(
        source=MODEL_SOURCE,
        savedir=MODEL_SAVEDIR,
        run_opts={"device": device},
    )
    _classifier_cache = classifier
    return classifier


def extract_logits(
    waveform_1d: torch.Tensor | np.ndarray,
    classifier=None,
) -> np.ndarray:
    """
    Run the classifier on one mono 16 kHz waveform.

    Returns
    -------
    np.ndarray of shape (K,) — log-probabilities (which work as logits for
    softmax-with-temperature).
    """
    if classifier is None:
        classifier = load_classifier()
    if isinstance(waveform_1d, np.ndarray):
        waveform_1d = torch.from_numpy(waveform_1d).float()
    if waveform_1d.ndim == 1:
        waveform_1d = waveform_1d.unsqueeze(0)
    out_prob, _, _, _ = classifier.classify_batch(waveform_1d)
    return out_prob.squeeze(0).detach().cpu().numpy()
