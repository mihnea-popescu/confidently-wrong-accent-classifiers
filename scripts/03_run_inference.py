"""
03_run_inference.py

Run the CommonAccent ECAPA model on every utterance in cal.csv and test.csv,
save scores to results/scores_cal.npz and results/scores_test.npz.

Outputs (per file):
  utterance_ids : (N,) int64
  scores        : (N, 16) float32  -- log-probabilities from the model
  durations_sec : (N,) float32     -- recomputed (used for HC3 verification)

Checkpointing
-------------
Inference can take a long time on CPU. We checkpoint every 200 utterances
to results/scores_<split>_partial.npz; if the script crashes or is killed,
re-running picks up where it left off. Final outputs are written only on
clean completion.

Device selection
----------------
Auto: CUDA > MPS (Apple Silicon) > CPU.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, Audio

from src.constants import (
    EDACC_HF_NAME,
    HF_CACHE_DIR,
    RESULTS_DIR,
    SPLITS_DIR,
    TARGET_SAMPLE_RATE,
    CV_LABELS,
)
from src.data import normalize_audio
from src.inference import load_classifier, extract_scores, select_device


CHECKPOINT_EVERY = 200


def run_split(split_name: str, df: pd.DataFrame, ds, classifier) -> dict:
    """
    Run inference on every row in df. Returns dict with utterance_ids,
    scores, durations_sec arrays.
    """
    n = len(df)
    print(f"\nRunning inference on '{split_name}' split: {n} utterances")

    # Resume from partial checkpoint if it exists
    partial_path = RESULTS_DIR / f"scores_{split_name}_partial.npz"
    final_path = RESULTS_DIR / f"scores_{split_name}.npz"

    if final_path.exists():
        print(f"  {final_path} already exists; skipping. Delete it to re-run.")
        return None

    done_ids = set()
    cached_scores = {}
    cached_durations = {}
    if partial_path.exists():
        print(f"  Found partial checkpoint, resuming...")
        partial = np.load(partial_path)
        for uid, scores, dur in zip(
            partial["utterance_ids"],
            partial["scores"],
            partial["durations_sec"],
        ):
            done_ids.add(int(uid))
            cached_scores[int(uid)] = scores
            cached_durations[int(uid)] = dur
        print(f"  Resumed: {len(done_ids)} of {n} already processed")

    utterance_ids: list[int] = []
    scores_list: list[np.ndarray] = []
    durations: list[float] = []

    start_time = time.time()
    last_checkpoint = start_time

    for i, row in enumerate(df.itertuples()):
        uid = int(row.utterance_id)
        if uid in done_ids:
            utterance_ids.append(uid)
            scores_list.append(cached_scores[uid])
            durations.append(float(cached_durations[uid]))
            continue

        # Load audio for this row
        ds_split_name = row.ds_split
        ds_index = int(row.ds_index)
        ex = ds[ds_split_name][ds_index]
        audio_field = ex["audio"]
        # When decode=True (default), HF returns {array, path, sampling_rate}
        arr = audio_field["array"]
        sr = audio_field["sampling_rate"]

        # Normalize: 16 kHz mono
        wave = normalize_audio(arr, sr)
        duration = len(wave) / TARGET_SAMPLE_RATE

        # Run model
        scores = extract_scores(wave, classifier=classifier)
        if scores.shape[0] != len(CV_LABELS):
            raise RuntimeError(
                f"Model output shape {scores.shape} does not match "
                f"expected {len(CV_LABELS)}. Check CV_LABELS in constants.py."
            )

        utterance_ids.append(uid)
        scores_list.append(scores.astype(np.float32))
        durations.append(float(duration))

        # Progress
        if (i + 1) % 50 == 0 or (i + 1) == n:
            elapsed = time.time() - start_time
            rate = (i + 1) / max(elapsed, 1e-9)
            eta = (n - (i + 1)) / max(rate, 1e-9)
            print(
                f"  [{i + 1}/{n}] {rate:.1f} utt/s  ETA {eta / 60:.1f} min"
            )

        # Checkpoint
        now = time.time()
        if (i + 1) % CHECKPOINT_EVERY == 0 or now - last_checkpoint > 300:
            np.savez(
                partial_path,
                utterance_ids=np.asarray(utterance_ids, dtype=np.int64),
                scores=np.stack(scores_list, axis=0),
                durations_sec=np.asarray(durations, dtype=np.float32),
            )
            last_checkpoint = now

    out = dict(
        utterance_ids=np.asarray(utterance_ids, dtype=np.int64),
        scores=np.stack(scores_list, axis=0),
        durations_sec=np.asarray(durations, dtype=np.float32),
    )
    np.savez(final_path, **out)
    print(f"  Wrote {final_path} (shape {out['scores'].shape})")

    # Clean up the partial checkpoint
    if partial_path.exists():
        partial_path.unlink()

    return out


def main():
    device = select_device()
    print(f"Device: {device}")

    cal_df = pd.read_csv(SPLITS_DIR / "cal.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    print(f"Loaded splits: cal={len(cal_df)}, test={len(test_df)}")

    print("Loading EdAcc (audio decoded for inference) ...")
    ds = load_dataset(EDACC_HF_NAME, cache_dir=str(HF_CACHE_DIR))
    # Audio is decoded by default; ensure it's the right type
    for split_name in ds.keys():
        ds[split_name] = ds[split_name].cast_column("audio", Audio(decode=True))

    print("Loading classifier ...")
    classifier = load_classifier(device=device)

    run_split("cal", cal_df, ds, classifier)
    run_split("test", test_df, ds, classifier)


if __name__ == "__main__":
    main()
