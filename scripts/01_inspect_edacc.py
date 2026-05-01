"""
01_inspect_edacc.py

Load EdAcc and print the L1 distribution. Run BEFORE committing the
l1_to_cv_mapping.md and BEFORE running any inference.

Output (saved to results/edacc_l1_distribution.json):
  {
    "L1_label": {"n_speakers": int, "n_utterances": int, "total_seconds": float},
    ...
  }
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset, Audio

from src.constants import EDACC_HF_NAME, RESULTS_DIR, HF_CACHE_DIR


def get_duration_seconds(audio_field, debug: bool = False) -> float:
    """
    Get duration of an audio example, handling all three storage cases:
      1. {path: "/disk/path.wav", ...}       — file on disk
      2. {bytes: b"...", path: None}         — inline bytes (parquet datasets)
      3. {array: np.ndarray, sampling_rate}  — already decoded
    """
    if audio_field is None:
        if debug:
            print("    [debug] audio_field is None")
        return 0.0

    if not isinstance(audio_field, dict):
        if debug:
            print(f"    [debug] audio_field is {type(audio_field).__name__}, not dict")
        return 0.0

    if debug:
        print(f"    [debug] audio_field keys: {list(audio_field.keys())}")
        for k, v in audio_field.items():
            if isinstance(v, bytes):
                print(f"      {k}: <bytes, len={len(v)}>")
            elif hasattr(v, "shape"):
                print(f"      {k}: <array, shape={v.shape}>")
            else:
                print(f"      {k}: {v!r}"[:120])

    import soundfile as sf

    # Case 1: file on disk
    path = audio_field.get("path")
    if path and isinstance(path, str) and Path(path).exists():
        try:
            return sf.info(path).duration
        except Exception as e:
            if debug:
                print(f"    [debug] sf.info(path) failed: {e}")

    # Case 2: inline bytes (most HF parquet-backed audio datasets)
    audio_bytes = audio_field.get("bytes")
    if audio_bytes:
        try:
            return sf.info(io.BytesIO(audio_bytes)).duration
        except Exception as e:
            if debug:
                print(f"    [debug] sf.info(BytesIO(bytes)) failed: {e}")

    # Case 3: already-decoded array
    arr = audio_field.get("array")
    sr = audio_field.get("sampling_rate")
    if arr is not None and sr:
        return len(arr) / sr

    if debug:
        print("    [debug] none of (path, bytes, array) yielded a duration")
    return 0.0


def main():
    print(f"Loading {EDACC_HF_NAME} ...")
    ds = load_dataset(EDACC_HF_NAME, cache_dir=str(HF_CACHE_DIR))

    print(f"\nDataset structure: {ds}")

    first_split_name = list(ds.keys())[0]

    # Disable audio decoding for the inspection walk — much faster.
    print("\nDisabling audio decoding for the inspection walk ...")
    for split_name in ds.keys():
        ds[split_name] = ds[split_name].cast_column("audio", Audio(decode=False))

    # Diagnostic: inspect the structure of the first audio example
    sample = ds[first_split_name][0]
    print(f"\nFirst example keys: {list(sample.keys())}")
    print("\nFirst example values (truncated):")
    for k, v in sample.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            print(f"  {k}: {v}")
        elif isinstance(v, dict):
            print(f"  {k}: <dict with keys: {list(v.keys())}>")
        else:
            print(f"  {k}: <{type(v).__name__}>")

    print("\n--- Diagnostic: inspecting first audio item ---")
    test_duration = get_duration_seconds(sample.get("audio"), debug=True)
    print(f"--- Duration of first item: {test_duration:.2f} seconds ---")

    if test_duration == 0.0:
        print("\n[!] Duration came out as 0 on the first item. Check the diagnostic")
        print("    output above. The L1 counts will still work, but total_seconds")
        print("    will be 0 across the board. Aborting before walking 19k examples.")
        return

    # Find L1 and speaker columns
    candidate_l1_keys = ["l1", "L1", "first_language", "native_language"]
    candidate_speaker_keys = ["speaker", "speaker_id", "spk_id", "client_id"]

    l1_key = next((k for k in candidate_l1_keys if k in sample), None)
    speaker_key = next((k for k in candidate_speaker_keys if k in sample), None)

    if l1_key is None or speaker_key is None:
        print(f"\n[!] Could not find required columns. l1_key={l1_key}, speaker_key={speaker_key}")
        return

    print(f"\nUsing L1 key: '{l1_key}'  speaker key: '{speaker_key}'")

    print("\nWalking dataset ...")
    stats = defaultdict(lambda: {"speakers": set(), "n_utterances": 0, "total_seconds": 0.0})

    total_examples = sum(len(s) for s in ds.values())
    seen = 0
    for split_name, split in ds.items():
        for ex in split:
            seen += 1
            if seen % 2000 == 0:
                print(f"  {seen}/{total_examples} ...")
            l1 = ex.get(l1_key) or "UNKNOWN"
            spk = ex.get(speaker_key) or "UNKNOWN"
            stats[l1]["speakers"].add(spk)
            stats[l1]["n_utterances"] += 1
            stats[l1]["total_seconds"] += get_duration_seconds(ex.get("audio"))

    out = {}
    for l1, s in sorted(stats.items(), key=lambda kv: -len(kv[1]["speakers"])):
        out[l1] = {
            "n_speakers": len(s["speakers"]),
            "n_utterances": s["n_utterances"],
            "total_seconds": round(s["total_seconds"], 1),
        }

    print(f"\n{'L1':<40} {'speakers':>10} {'utterances':>12} {'hours':>8}")
    print("-" * 75)
    for l1, info in out.items():
        hours = info["total_seconds"] / 3600.0
        print(
            f"{l1[:40]:<40} {info['n_speakers']:>10} "
            f"{info['n_utterances']:>12} {hours:>8.2f}"
        )

    total_hours = sum(v["total_seconds"] for v in out.values()) / 3600.0
    total_utts = sum(v["n_utterances"] for v in out.values())
    total_speakers = sum(v["n_speakers"] for v in out.values())
    print("-" * 75)
    print(
        f"{'TOTAL':<40} {total_speakers:>10} "
        f"{total_utts:>12} {total_hours:>8.2f}"
    )

    out_path = RESULTS_DIR / "edacc_l1_distribution.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()