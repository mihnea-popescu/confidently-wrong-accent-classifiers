"""
Single source of truth for labels, paths, seeds, and pinned versions.
"""

from pathlib import Path

# Reproducibility
SEED = 42

# Paths (everything relative to repo root)
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
SPLITS_DIR = REPO_ROOT / "splits"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
PRETRAINED_DIR = REPO_ROOT / "pretrained_models"
HF_CACHE_DIR = REPO_ROOT / "hf_cache"

# Make sure output dirs exist
for d in (SPLITS_DIR, RESULTS_DIR, FIGURES_DIR, PRETRAINED_DIR, HF_CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Model
MODEL_SOURCE = "Jzuluaga/accent-id-commonaccent_ecapa"
MODEL_REVISION = None  # FILL IN: HuggingFace commit hash after first download
MODEL_SAVEDIR = str(PRETRAINED_DIR / "accent-id-commonaccent_ecapa")

# 16 CV labels the model outputs.
# Sourced from the model's label_encoder.txt at download time.
# DO NOT edit by hand — verify against the downloaded file in 00_smoke_test.py.
# The order here MUST match the model's logit ordering.
CV_LABELS = [
    "us",
    "england",
    "australia",
    "indian",
    "canada",
    "bermuda",
    "scotland",
    "african",
    "ireland",
    "newzealand",
    "wales",
    "malaysia",
    "philippines",
    "singapore",
    "hongkong",
    "southatlandtic",  # NB: typo is in the model, do NOT correct
]

# Dataset
EDACC_HF_NAME = "edinburghcstr/edacc"

# Audio
TARGET_SAMPLE_RATE = 16_000  # Hz, mono

# Inclusion thresholds (per prereg)
MIN_SPEAKERS_PER_GROUP = 4
MIN_UTTERANCES_PER_GROUP = 100
MIN_UTTERANCE_DURATION_SEC = 1.0  # drop turns under this

# Calibration
ECE_BINS = 15
LBFGS_MAX_ITER = 50
TEMPERATURE_INIT = 1.0
SELECTIVE_ACCURACY_COVERAGE = 0.80

# Decision rule (per prereg)
WORST_GROUP_ECE_RELATIVE_REDUCTION_TARGET = 0.30  # 30% reduction
BEST_GROUP_ECE_RELATIVE_INCREASE_LIMIT = 0.20     # max 20% increase
