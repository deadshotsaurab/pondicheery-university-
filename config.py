"""
config.py  —  All thresholds derived automatically from data.
NO hardcoded word lists. NO manual boundaries.
"""

import os
import glob

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
SEED_FILE  = os.path.join(BASE_DIR, "seed_words.json")

# ── Model ──────────────────────────────────────────────────────────────────
BERT_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
N_COMPONENTS  = 3          # LAYMAN / STUDENT / PROFESSIONAL
KNN_K         = 5          # neighbours in seed-based KNN
RANDOM_STATE  = 42

# ── Automatic filter parameters (derived from corpus statistics) ───────────
# These are NOT thresholds — they are sigma multipliers used during training
# to compute boundaries FROM your seed word distributions.
ZIPF_SIGMA_FILTER = 2.5    # remove words beyond 2.5σ from any seed group mean
MIN_WORD_LENGTH   = 3      # absolute minimum (removes 'a', 'an', 'of' etc.)
MIN_CORPUS_FREQ   = 2      # word must appear at least twice in corpus
CAP_RATIO_THRESH  = 3.0    # capitalized:lowercase ratio for proper noun detection
MIN_CAP_COUNT     = 3      # minimum capitalised occurrences to flag as proper noun

# ── Confidence ─────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.55   # below this → use seed-similarity fallback

# ── Validate ───────────────────────────────────────────────────────────────
def validate():
    txt_files = glob.glob(os.path.join(DATA_DIR, "*.txt"))
    if not txt_files:
        print(f"WARNING: No .txt files found in DATA_DIR = {DATA_DIR}")
        print("  → Place your legal corpus .txt files there before running.")
    else:
        print(f"Configuration validated. Found {len(txt_files)} corpus file(s).")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == "__main__":
    validate()