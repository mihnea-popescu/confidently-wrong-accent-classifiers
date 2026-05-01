"""
08_select_qualitative.py

Deterministically pick 8-10 test-split utterances for the paper's qualitative
analysis section. Selection is committed to a CSV before listening, so the
choices aren't influenced by what the listening reveals.

Per the prereg: this happens AFTER quantitative analysis (so we know which
groups are interesting) but BEFORE writing the qualitative section (so we
write honestly about what we hear).

Selection strategy (10 utterances total):
    - us: 1 correctly classified, 1 misclassified — the model's "best" group
    - england: 1 correctly classified
    - indian: 1 correctly classified — the partial-improvement case
    - scotland: 1 — the noisy small-group case
    - african: 1 misclassified — accuracy ~1%, where does the model go wrong
    - ireland: 1 misclassified — same as african
    - Vietnamese (OOV): 2 — what does the model say when the L1 isn't in its labels
    - 1 short utterance (<2s) from any group — investigate HC3 failure

All selections use a fixed RNG seed for reproducibility.

Outputs:
    paper/qualitative_selection.csv
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import soundfile as sf
from datasets import load_dataset, Audio

from src.constants import (
    CV_LABELS,
    EDACC_HF_NAME,
    HF_CACHE_DIR,
    LOGIT_SCALE,
    REPO_ROOT,
    RESULTS_DIR,
    SEED,
    SPLITS_DIR,
)


QUALITATIVE_SEED = SEED + 1  # different stream from the cal/test split RNG
N_UTTERANCES = 10

# Output directory for extracted audio clips so you can listen
LISTEN_DIR = REPO_ROOT / "qualitative_audio"
LISTEN_DIR.mkdir(parents=True, exist_ok=True)


def softmax_with_scale(scores: np.ndarray, scale: float = LOGIT_SCALE) -> np.ndarray:
    z = scale * scores
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def pick_one(
    df: pd.DataFrame,
    rng: np.random.Generator,
    description: str,
) -> dict | None:
    """Pick a single row from a filtered dataframe. Returns None if empty."""
    if len(df) == 0:
        print(f"  [!] No utterances available for: {description}")
        return None
    row = df.sample(n=1, random_state=int(rng.integers(0, 2**31))).iloc[0]
    return dict(
        utterance_id=int(row["utterance_id"]),
        ds_split=row["ds_split"],
        ds_index=int(row["ds_index"]),
        speaker=row["speaker"],
        group=row["group"],
        track=row["track"],
        target_label=row.get("target_label", ""),
        target_idx=int(row["target_idx"]),
        duration_sec=float(row["duration_sec"]),
        predicted_label=row["predicted_label"],
        predicted_confidence=float(row["predicted_confidence"]),
        correct=bool(row["correct"]),
        selection_reason=description,
    )


def main():
    # Load test split + scores, compute per-utterance prediction
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    npz = np.load(RESULTS_DIR / "scores_test.npz")
    inf_df = pd.DataFrame({"utterance_id": npz["utterance_ids"]})
    test_df = test_df.merge(inf_df, on="utterance_id", how="inner")
    id_to_idx = {int(uid): i for i, uid in enumerate(npz["utterance_ids"])}
    perm = np.array([int(id_to_idx[int(u)]) for u in test_df["utterance_id"]])
    raw_scores = npz["scores"][perm]
    probs = softmax_with_scale(raw_scores)

    pred_idx = probs.argmax(axis=1)
    pred_conf = probs.max(axis=1)
    pred_label = [CV_LABELS[i] for i in pred_idx]

    test_df = test_df.copy()
    test_df["predicted_idx"] = pred_idx
    test_df["predicted_label"] = pred_label
    test_df["predicted_confidence"] = pred_conf
    # correctness only meaningful for Track A
    test_df["correct"] = (
        (test_df["track"] == "A") & (test_df["predicted_idx"] == test_df["target_idx"])
    )

    rng = np.random.default_rng(QUALITATIVE_SEED)
    selections: list[dict] = []

    # Track A: one correct + one wrong us, plus one correct each for england/indian/scotland
    selections.append(pick_one(
        test_df[(test_df.group == "us") & test_df.correct],
        rng, "us: correctly classified — best in-vocab case",
    ))
    selections.append(pick_one(
        test_df[(test_df.group == "us") & ~test_df.correct],
        rng, "us: misclassified — what does the model fall back to",
    ))
    selections.append(pick_one(
        test_df[(test_df.group == "england") & test_df.correct],
        rng, "england: correctly classified",
    ))
    selections.append(pick_one(
        test_df[(test_df.group == "indian") & test_df.correct],
        rng, "indian: correctly classified — partial-improvement case",
    ))
    selections.append(pick_one(
        test_df[test_df.group == "scotland"],
        rng, "scotland: any — noisy small-group case",
    ))

    # african and ireland: accuracy ~1%, so any sample is almost certainly misclassified
    selections.append(pick_one(
        test_df[test_df.group == "african"],
        rng, "african: where does the model send Nigerian English",
    ))
    selections.append(pick_one(
        test_df[test_df.group == "ireland"],
        rng, "ireland: where does the model send Irish English",
    ))

    # Two Vietnamese OOV speakers
    viet = test_df[test_df.group == "Vietnamese"]
    if len(viet) >= 2:
        sample = viet.sample(n=2, random_state=int(rng.integers(0, 2**31)))
        for _, row in sample.iterrows():
            selections.append(dict(
                utterance_id=int(row["utterance_id"]),
                ds_split=row["ds_split"],
                ds_index=int(row["ds_index"]),
                speaker=row["speaker"],
                group=row["group"],
                track=row["track"],
                target_label="",
                target_idx=int(row["target_idx"]),
                duration_sec=float(row["duration_sec"]),
                predicted_label=row["predicted_label"],
                predicted_confidence=float(row["predicted_confidence"]),
                correct=False,
                selection_reason="Vietnamese OOV: where does the model fall back",
            ))

    # One short utterance (<2s) from any Track A group, to investigate HC3 failure
    short_track_a = test_df[(test_df.duration_sec < 2.0) & (test_df.track == "A")]
    selections.append(pick_one(
        short_track_a, rng,
        "short utterance (<2s, Track A): HC3 investigation",
    ))

    selections = [s for s in selections if s is not None]
    print(f"Selected {len(selections)} utterances")

    out_df = pd.DataFrame(selections)
    out_path = REPO_ROOT / "paper" / "qualitative_selection.csv"
    out_path.parent.mkdir(exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote selection to {out_path}")

    # Extract the actual audio clips so you can listen.
    print("\nExtracting audio clips for listening ...")
    ds = load_dataset(EDACC_HF_NAME, cache_dir=str(HF_CACHE_DIR))
    for split in ds.keys():
        ds[split] = ds[split].cast_column("audio", Audio(decode=False))

    listen_paths: list[str] = []
    for sel in selections:
        ex = ds[sel["ds_split"]][sel["ds_index"]]
        audio_field = ex["audio"]
        # Get audio bytes (the inline-bytes case for HF parquet datasets)
        if isinstance(audio_field, dict):
            ab = audio_field.get("bytes")
            path = audio_field.get("path")
            if ab:
                # Read with soundfile from bytes
                wav, sr = sf.read(io.BytesIO(ab))
            elif path and Path(path).exists():
                wav, sr = sf.read(path)
            else:
                listen_paths.append("")
                continue
        else:
            listen_paths.append("")
            continue

        # Write a wav file with a short, descriptive name
        fname = (
            f"{sel['utterance_id']:05d}"
            f"__{sel['group']}"
            f"__pred-{sel['predicted_label']}"
            f"__{'OK' if sel['correct'] else 'X'}"
            f".wav"
        )
        out_wav = LISTEN_DIR / fname
        sf.write(out_wav, wav, sr)
        # Store path relative to repo root so the CSV stays portable
        listen_paths.append(str(out_wav.relative_to(REPO_ROOT)))
        print(f"  {fname}")

    out_df["listen_path"] = listen_paths
    out_df["listening_note"] = ""  # to be filled in by you
    out_df.to_csv(out_path, index=False)
    print(f"\nUpdated {out_path} with listen_path and empty listening_note column.")
    print(f"Audio clips written to: {LISTEN_DIR}/")


if __name__ == "__main__":
    main()