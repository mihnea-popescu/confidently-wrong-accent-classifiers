"""
02_make_splits.py

Produce speaker-stratified calibration and test splits from EdAcc, applying
the L1-to-CV mapping committed in paper/l1_to_cv_mapping.md.

Outputs
-------
splits/cal.csv   - one row per utterance in the calibration split
splits/test.csv  - one row per utterance in the test split

Schema (both files):
  utterance_id   - internal int ID, unique across the project
  ds_split       - "validation" or "test" (the EdAcc split this row came from)
  ds_index       - row index within that EdAcc split (for loading audio)
  speaker        - speaker ID from EdAcc
  l1             - original L1 string from EdAcc
  group          - normalized group label (e.g. "us", "Vietnamese", "OOV-agg")
  track          - "A" (in-vocab) or "B" (OOV) or "excluded"
  target_label   - CV label string (Track A only, else empty)
  target_idx     - index in CV_LABELS (Track A only, else -1)
  duration_sec   - from soundfile.info()
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import soundfile as sf
from datasets import load_dataset, Audio

from src.constants import (
    CV_LABELS,
    EDACC_HF_NAME,
    HF_CACHE_DIR,
    MIN_SPEAKERS_PER_GROUP,
    MIN_UTTERANCES_PER_GROUP,
    MIN_UTTERANCE_DURATION_SEC,
    SEED,
    SPLITS_DIR,
)
from src.data import speaker_level_split


# ----------------------------------------------------------------------
# L1 -> (track, group, target_label) mapping
# ----------------------------------------------------------------------
# Track A: maps to a CV label
TRACK_A_MAPPING = {
    "Southern British English": "england",
    "Mainstream US English": "us",
    "Irish English": "ireland",
    "Nigerian English": "african",
    "Indian English": "indian",
    "Scottish English": "scotland",
}

# Track B: out-of-vocabulary L1s with sufficient speakers (kept as own group)
TRACK_B_GROUPS = {
    "Vietnamese",
    "Spanish",
    "Italian",
    "Mandarin",
    "Bulgarian",
    "Catalan",
}

# Excluded-with-context (no CV mapping; documented in the mapping file)
EXCLUDED_NAMED = {
    "Jamaican English",          # no clean CV mapping
    "Kenyan English",            # would conflate African L1s
    "Ghanain English",
    "South African English",
    "Hindi", "Urdu", "Sinhalese",  # L2 South Asian; conflation risk
}


def classify_row(l1: str) -> tuple[str, str, str, int]:
    """
    Return (track, group, target_label, target_idx) for a given L1.

    For excluded rows we still emit the row but track="excluded" so the
    caller can drop them deterministically later.
    """
    if l1 in TRACK_A_MAPPING:
        cv_label = TRACK_A_MAPPING[l1]
        if cv_label not in CV_LABELS:
            raise RuntimeError(
                f"Mapping target '{cv_label}' is not in CV_LABELS. "
                f"Check src/constants.py vs the model's label_encoder.txt."
            )
        return ("A", cv_label, cv_label, CV_LABELS.index(cv_label))

    if l1 in TRACK_B_GROUPS:
        return ("B", l1, "", -1)

    if l1 in EXCLUDED_NAMED:
        return ("excluded", l1, "", -1)

    # Below threshold — pool into OOV-aggregate for the H2 cross-track
    # confidence comparison, but don't use them for per-group ECE.
    return ("B", "OOV-aggregate", "", -1)


def get_duration(audio_field) -> float:
    if audio_field is None or not isinstance(audio_field, dict):
        return 0.0
    path = audio_field.get("path")
    if path and isinstance(path, str) and Path(path).exists():
        try:
            return sf.info(path).duration
        except Exception:
            pass
    audio_bytes = audio_field.get("bytes")
    if audio_bytes:
        try:
            return sf.info(io.BytesIO(audio_bytes)).duration
        except Exception:
            return 0.0
    arr = audio_field.get("array")
    sr = audio_field.get("sampling_rate")
    if arr is not None and sr:
        return len(arr) / sr
    return 0.0


def main():
    print(f"Loading {EDACC_HF_NAME} ...")
    ds = load_dataset(EDACC_HF_NAME, cache_dir=str(HF_CACHE_DIR))

    # No audio decoding for the metadata walk
    for split_name in ds.keys():
        ds[split_name] = ds[split_name].cast_column("audio", Audio(decode=False))

    rows = []
    next_id = 0
    for split_name, split in ds.items():
        for ds_index, ex in enumerate(split):
            l1 = ex.get("l1") or "UNKNOWN"
            track, group, target_label, target_idx = classify_row(l1)
            duration = get_duration(ex.get("audio"))
            rows.append(
                dict(
                    utterance_id=next_id,
                    ds_split=split_name,
                    ds_index=ds_index,
                    speaker=ex.get("speaker") or "UNKNOWN",
                    l1=l1,
                    group=group,
                    track=track,
                    target_label=target_label,
                    target_idx=target_idx,
                    duration_sec=round(duration, 3),
                )
            )
            next_id += 1
            if next_id % 2000 == 0:
                print(f"  {next_id} rows ...")

    df = pd.DataFrame(rows)
    print(f"\nTotal rows before filtering: {len(df)}")

    # Drop excluded rows entirely from cal/test files (they're documented
    # in the mapping file; we don't carry them through inference)
    df = df[df["track"] != "excluded"].copy()
    print(f"After dropping 'excluded' track:  {len(df)}")

    # Drop too-short utterances (per prereg)
    n_before = len(df)
    df = df[df["duration_sec"] >= MIN_UTTERANCE_DURATION_SEC].copy()
    print(
        f"After dropping <{MIN_UTTERANCE_DURATION_SEC}s utterances: "
        f"{len(df)} (-{n_before - len(df)})"
    )

    # Sanity check: every Track A group should have ≥ MIN_SPEAKERS_PER_GROUP
    # speakers and ≥ MIN_UTTERANCES_PER_GROUP utterances. If not, something
    # is wrong with the mapping.
    print("\nGroup sizes after filtering:")
    print(f"  {'group':<22} {'track':>6} {'speakers':>10} {'utts':>8} {'hours':>8}")
    track_a_failed = []
    for grp, sub in df.groupby("group"):
        n_spk = sub["speaker"].nunique()
        n_utt = len(sub)
        hrs = sub["duration_sec"].sum() / 3600.0
        track = sub["track"].iloc[0]
        ok = n_spk >= MIN_SPEAKERS_PER_GROUP and n_utt >= MIN_UTTERANCES_PER_GROUP
        marker = "" if ok else "   <-- below threshold"
        print(f"  {grp:<22} {track:>6} {n_spk:>10} {n_utt:>8} {hrs:>8.2f}{marker}")
        if track == "A" and not ok:
            track_a_failed.append(grp)

    if track_a_failed:
        print(
            f"\n[!] Track A groups below threshold: {track_a_failed}\n"
            "    The mapping commits to these groups; investigate before proceeding."
        )

    # Speaker-level split, stratified by group (for Track A and named Track B)
    # The OOV-aggregate group is split with the others on its own.
    # Per the prereg: 50% test / 50% cal at the speaker level.
    cal_df, test_df = speaker_level_split(
        df, speaker_col="speaker", group_col="group",
        test_fraction=0.5, seed=SEED,
    )

    # Annotate which split each row belongs to
    cal_df = cal_df.assign(split="cal")
    test_df = test_df.assign(split="test")

    cal_path = SPLITS_DIR / "cal.csv"
    test_path = SPLITS_DIR / "test.csv"
    cal_df.to_csv(cal_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"\nWrote {cal_path} ({len(cal_df)} rows)")
    print(f"Wrote {test_path} ({len(test_df)} rows)")

    # Per-group cal/test breakdown for sanity
    print("\nPer-group split sizes:")
    print(f"  {'group':<22} {'cal_spk':>8} {'cal_utt':>8} {'test_spk':>9} {'test_utt':>9}")
    for grp in sorted(df["group"].unique()):
        c = cal_df[cal_df["group"] == grp]
        t = test_df[test_df["group"] == grp]
        print(
            f"  {grp:<22} {c['speaker'].nunique():>8} {len(c):>8} "
            f"{t['speaker'].nunique():>9} {len(t):>9}"
        )

    # Save a small manifest with summary stats for the paper
    manifest = {
        "seed": SEED,
        "min_utterance_duration_sec": MIN_UTTERANCE_DURATION_SEC,
        "min_speakers_per_group": MIN_SPEAKERS_PER_GROUP,
        "min_utterances_per_group": MIN_UTTERANCES_PER_GROUP,
        "n_total_rows_after_filter": int(len(df)),
        "n_cal": int(len(cal_df)),
        "n_test": int(len(test_df)),
        "groups": {},
    }
    for grp, sub in df.groupby("group"):
        manifest["groups"][grp] = {
            "track": sub["track"].iloc[0],
            "n_speakers": int(sub["speaker"].nunique()),
            "n_utterances": int(len(sub)),
            "total_seconds": float(round(sub["duration_sec"].sum(), 1)),
        }
    (SPLITS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written to {SPLITS_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
