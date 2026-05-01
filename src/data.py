"""
EdAcc data loading, speaker-level splitting, and audio normalization.

Audio is resampled to 16 kHz mono. Utterances under MIN_UTTERANCE_DURATION_SEC
are dropped per the prereg.

Speaker-level splitting: a given speaker appears in either 'cal' or 'test',
never both. Without this, temperature fitting overfits to per-speaker quirks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torchaudio
from typing import Iterable

from src.constants import (
    TARGET_SAMPLE_RATE,
    MIN_UTTERANCE_DURATION_SEC,
    SEED,
)


def normalize_audio(
    waveform: torch.Tensor | np.ndarray,
    sample_rate: int,
) -> torch.Tensor:
    """
    Take an audio array of any shape/sample-rate and return a 1D tensor at
    TARGET_SAMPLE_RATE in mono.
    """
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform).float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # -> (1, samples)
    if waveform.shape[0] > 1:
        # Multi-channel -> mono
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, TARGET_SAMPLE_RATE
        )
    return waveform.squeeze(0)  # 1D


def utterance_duration_seconds(audio_dict: dict) -> float:
    arr = audio_dict.get("array")
    sr = audio_dict.get("sampling_rate")
    if arr is None or not sr:
        return 0.0
    return len(arr) / sr


def speaker_level_split(
    df: pd.DataFrame,
    speaker_col: str = "speaker_id",
    group_col: str = "l1_group",
    test_fraction: float = 0.5,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame so each speaker appears in only one of (cal, test).
    Stratified by L1 group: each group contributes its share to both splits.

    Parameters
    ----------
    df : DataFrame with at least `speaker_col` and `group_col` columns.
    speaker_col : column identifying the speaker
    group_col : column identifying the group to stratify by
    test_fraction : fraction of speakers per group assigned to test split
    seed : RNG seed

    Returns
    -------
    (cal_df, test_df) tuple of DataFrames

    Notes
    -----
    With small group sizes (4-6 speakers per L1), the split won't be
    perfectly balanced; we round down for the test set and the rest go to
    cal. If a group has only 1 speaker, that speaker goes to test (we can't
    fit per-group T anyway, but at least the group appears in evaluation).
    """
    rng = np.random.RandomState(seed)
    cal_speakers = []
    test_speakers = []

    for group, sub in df.groupby(group_col):
        speakers = sub[speaker_col].unique()
        rng.shuffle(speakers)
        n = len(speakers)
        if n == 1:
            test_speakers.extend(speakers.tolist())
            continue
        n_test = max(1, int(np.floor(n * test_fraction)))
        test_speakers.extend(speakers[:n_test].tolist())
        cal_speakers.extend(speakers[n_test:].tolist())

    cal_df = df[df[speaker_col].isin(cal_speakers)].copy()
    test_df = df[df[speaker_col].isin(test_speakers)].copy()

    # Sanity: no speaker in both
    overlap = set(cal_df[speaker_col]) & set(test_df[speaker_col])
    assert not overlap, f"Speaker leak: {overlap}"

    return cal_df, test_df


def filter_by_duration(
    durations_sec: Iterable[float],
    min_sec: float = MIN_UTTERANCE_DURATION_SEC,
) -> np.ndarray:
    """Boolean mask of utterances >= min_sec."""
    return np.asarray(list(durations_sec)) >= min_sec
