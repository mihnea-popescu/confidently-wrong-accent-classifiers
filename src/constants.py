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
    'england',
    'us',
    'canada',
    'australia',
    'indian',
    'scotland',
    'ireland',
    'african',
    'malaysia',
    'newzealand',
    'southatlandtic', # Not a typo
    'bermuda',
    'philippines',
    'hongkong',
    'wales',
    'singapore'
]

# ---- Cosine -> logit scale factor ----
#
# The CommonAccent ECAPA-TDNN model uses a cosine-similarity classifier
# (L2-normalized embeddings · L2-normalized class prototypes). Inference
# returns raw cosines in [-1, 1].
#
# At training time, the AAM-softmax loss applies a fixed scale factor S
# before softmax so that the loss has useful gradients on a near-uniform
# distribution of cosines. The SpeechBrain default for this recipe family
# is S = 30. The model was trained to make `softmax(S * cos)` peaky on the
# correct class.
#
# At inference, SpeechBrain returns raw cosines without applying S. So our
# analysis must apply it before computing probabilities or fitting
# temperature scaling. Without S, any softmax over cosines comes out
# near-uniform and ECE is meaningless.
#
# If the actual training scale differed from 30, the fitted temperature
# will absorb the discrepancy (T near 1.0 means our scale assumption is
# correct; T far from 1.0 may indicate a mismatch — note for limitations).
LOGIT_SCALE = 30.0

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
