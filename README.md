# Confidently Wrong: Group-Conditional Miscalibration in Accent Classifiers

Audit and intervention on the calibration of an off-the-shelf English accent
classifier across speakers from different first-language (L1) backgrounds,
using EdAcc as the evaluation set. AIES-style final project.

## Status

Pre-registration committed as `prereg-v1`. See `PREREGISTRATION.md`.

## Layout

```
.
├── PREREGISTRATION.md   pre-registered protocol (frozen before evaluation)
├── requirements.txt        pinned dependencies
├── src/
│   ├── constants.py        label list, paths, seeds
│   ├── data.py             EdAcc loader + speaker-level splitting
│   ├── inference.py        model loading + logit extraction
│   ├── metrics.py          ECE, Brier, selective accuracy
│   ├── temperature.py      single-T and group-conditional T fitting
│   └── plots.py            reliability diagrams
├── scripts/
│   ├── 00_smoke_test.py    hello-world: load model, run on one clip
│   ├── 01_inspect_edacc.py inspect L1 distribution before any inference
│   ├── 02_make_splits.py   speaker-level split, save CSVs
│   ├── 03_run_inference.py inference on calibration + test, save logits
│   ├── 04_baseline_eval.py per-group metrics on baseline
│   ├── 05_fit_temperature.py
│   └── 06_intervention_eval.py
├── splits/                 committed split CSVs (reproducible)
├── results/                committed JSON/CSV outputs from each phase
├── figures/                generated plots
├── paper/
│   ├── main.md             the paper
│   └── l1_to_cv_mapping.md committed before inference
└── tests/
    └── test_metrics.py     unit tests for ECE
```

## Quickstart

```bash
conda create -n confidently-wrong python=3.11 &&
conda activate confidently-wrong &&
pip install -r requirements.txt
```

Run scripts in order; each one writes to `results/` and reads what previous
scripts wrote. Splits are written to `splits/` and committed to the repo
for reproducibility.

## Reproducibility

- Random seed: 42, set in `src/constants.py`
- Splits: speaker-level, stratified by L1, written to `splits/*.csv`
- Model revision: pinned in `src/constants.py`
- All numerical results: written to `results/*.json` with timestamps

## Citation

The model is from Zuluaga-Gomez et al., "CommonAccent: Exploring Large
Acoustic Pretrained Models for Accent Classification Based on Common Voice"
(Interspeech 2023, arXiv:2305.18283). The dataset is from Sanabria et al.,
"The Edinburgh International Accents of English Corpus" (ICASSP 2023,
arXiv:2303.18110).
