"""
feature_engineering.py
All filters derived automatically from corpus statistics and seed word
distributions. Zero hardcoded word lists or manual thresholds.
"""

import re
import os
import glob
import json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from wordfreq import zipf_frequency

import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer

import pyphen
import config
from seed_similarity import SeedSimilarityFeaturizer

# ── Ensure NLTK data is available ─────────────────────────────────────────
def _ensure_nltk():
    for pkg in ["stopwords", "wordnet", "omw-1.4"]:
        try:
            if pkg == "stopwords":
                nltk_stopwords.words("english")
            elif pkg == "wordnet":
                from nltk.corpus import wordnet
                wordnet.synsets("test")
        except LookupError:
            nltk.download(pkg, quiet=True)

_ensure_nltk()

# ── Stopwords from NLTK (research-standard corpus, not manually typed) ────
_NLTK_STOPS = set(nltk_stopwords.words("english"))

# ── Hyphenation for syllable counting ─────────────────────────────────────
_HYPHENATOR = pyphen.Pyphen(lang="en")
_LEMMATIZER = WordNetLemmatizer()


# ═══════════════════════════════════════════════════════════════════════════
# TEXT READING
# ═══════════════════════════════════════════════════════════════════════════

def read_corpus(data_dir: str):
    """Read all .txt files. Returns (list_of_doc_strings, word_freq_counter)."""
    paths = glob.glob(os.path.join(data_dir, "*.txt"))
    if not paths:
        raise FileNotFoundError(f"No .txt files found in {data_dir}")

    documents, all_tokens = [], []
    for p in paths:
        try:
            text = open(p, encoding="utf-8", errors="ignore").read()
        except Exception:
            text = open(p, encoding="latin-1", errors="ignore").read()
        documents.append(text)
        tokens = re.findall(r"[a-zA-Z]+", text)
        all_tokens.extend(tokens)

    print(f"STEP 1: Read {len(documents)} document(s).")
    return documents, Counter(all_tokens)


# ═══════════════════════════════════════════════════════════════════════════
# AUTOMATIC PROPER NOUN DETECTION
# Uses corpus statistics only — no name lists
# ═══════════════════════════════════════════════════════════════════════════

def detect_proper_nouns(documents: list, cap_ratio: float, min_cap: int) -> set:
    """
    Flag words that appear capitalised (mid-sentence) much more often than
    lowercase. This catches names and places automatically from the corpus.
    Ratio and minimum count come from config — not hardcoded here.
    """
    cap_count   = Counter()   # "Singh", "Bombay"
    lower_count = Counter()   # "bombay", "singh"

    for doc in documents:
        sentences = re.split(r"(?<=[.!?])\s+", doc)
        for sent in sentences:
            words = re.findall(r"[A-Za-z]+", sent)
            for i, w in enumerate(words):
                if i == 0:          # first word of sentence → skip
                    continue
                if w[0].isupper():
                    cap_count[w.lower()] += 1
                else:
                    lower_count[w.lower()] += 1

    flagged = set()
    for word, cc in cap_count.items():
        lc = lower_count.get(word, 0)
        if cc >= min_cap and (lc == 0 or cc / (lc + 1) >= cap_ratio):
            flagged.add(word)

    print(f"  Auto proper-noun filter: {len(flagged)} words flagged "
          f"(cap_ratio≥{cap_ratio}, min_cap≥{min_cap})")
    return flagged


# ═══════════════════════════════════════════════════════════════════════════
# AUTOMATIC ZIPF BOUNDARY DETECTION
# Boundaries derived from seed word statistics — not manually set
# ═══════════════════════════════════════════════════════════════════════════

def compute_zipf_bounds_from_seeds(seed_words: dict, sigma: float):
    """
    Compute corpus-filtering zipf bounds from the seed word distributions.

    Upper bound = mean_zipf(LAYMAN seeds) + sigma * std_zipf(ALL seeds)
    Lower bound = mean_zipf(PROFESSIONAL seeds) - sigma * std_zipf(ALL seeds)

    Words outside [lower, upper] are statistical outliers relative to the
    seed distributions — they are likely OCR noise or ultra-stopwords.
    """
    all_seed_zipfs = []
    group_zipfs    = {}

    for label, words in seed_words.items():
        zs = [zipf_frequency(w, "en") for w in words if zipf_frequency(w, "en") > 0]
        if zs:
            group_zipfs[label] = zs
            all_seed_zipfs.extend(zs)

    if not all_seed_zipfs:
        return 1.0, 6.0   # safe fallback

    global_std = float(np.std(all_seed_zipfs))

    layman_mean = float(np.mean(group_zipfs.get("layman", all_seed_zipfs)))
    prof_mean   = float(np.mean(group_zipfs.get("professional", all_seed_zipfs)))

    upper = layman_mean + sigma * global_std
    lower = max(0.0, prof_mean - sigma * global_std)

    print(f"  Auto zipf bounds: [{lower:.2f}, {upper:.2f}]  "
          f"(seed-derived, σ={sigma})")
    return lower, upper


# ═══════════════════════════════════════════════════════════════════════════
# WORD FILTERING
# ═══════════════════════════════════════════════════════════════════════════

def filter_words(word_freq: Counter,
                 documents: list,
                 seed_words: dict,
                 min_freq: int,
                 min_len: int,
                 sigma: float,
                 cap_ratio: float,
                 min_cap: int) -> Counter:
    """
    Remove noise words automatically.  All thresholds come from parameters
    computed from corpus / seed statistics — nothing is hardcoded here.
    """
    before = len(word_freq)

    # 1. NLTK stopwords (research-standard list, not manually written)
    stops = _NLTK_STOPS

    # 2. Auto proper noun detection from corpus statistics
    proper_nouns = detect_proper_nouns(documents, cap_ratio, min_cap)

    # 3. Zipf bounds from seed distributions
    zipf_lo, zipf_hi = compute_zipf_bounds_from_seeds(seed_words, sigma)

    removed = defaultdict(int)
    kept    = Counter()

    for word, freq in word_freq.items():
        w = word.lower()

        if len(w) < min_len:
            removed["too_short"] += 1
            continue
        if not w.isalpha():
            removed["non_alpha"] += 1
            continue
        if freq < min_freq:
            removed["too_rare(freq)"] += 1
            continue
        if w in stops:
            removed["stopword(nltk)"] += 1
            continue
        if w in proper_nouns:
            removed["proper_noun"] += 1
            continue

        z = zipf_frequency(w, "en")
        if z > zipf_hi:
            removed["too_common(zipf)"] += 1
            continue
        if 0 < z < zipf_lo:
            removed["too_rare(zipf)"] += 1
            continue

        kept[w] = freq

    after = len(kept)
    print(f"  filter_words: {before} → {after} words kept")
    for reason, count in sorted(removed.items(), key=lambda x: -x[1]):
        print(f"    removed {count:5d} [{reason}]")

    return kept


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def _syllable_count(word: str) -> int:
    parts = _HYPHENATOR.inserted(word)
    return max(1, parts.count("-") + 1) if parts else max(1, len(word) // 3)


def _wordnet_depth(word: str) -> float:
    try:
        from nltk.corpus import wordnet
        synsets = wordnet.synsets(word)
        if not synsets:
            return 0.0
        depths = []
        for s in synsets[:3]:
            paths = s.hypernym_paths()
            if paths:
                depths.append(max(len(p) for p in paths))
        return float(np.mean(depths)) if depths else 0.0
    except Exception:
        return 0.0


def _domain_specificity(word: str, word_freq: Counter, total_tokens: int) -> float:
    """
    TF-IDF-inspired domain score: how much more this word appears in the
    legal corpus vs. general English frequency.
    """
    corpus_tf = word_freq.get(word, 0) / max(total_tokens, 1)
    general_freq = max(zipf_frequency(word, "en"), 0.01)
    # Convert zipf to probability: 10^(zipf - 9)
    general_prob = 10 ** (general_freq - 9)
    if general_prob <= 0:
        return 0.0
    return float(np.log1p(corpus_tf / general_prob))


def extract_all_features(word_freq: Counter,
                         documents: list,
                         seed_words: dict) -> pd.DataFrame:
    """
    Extract all features automatically. Returns DataFrame with one row per word.
    """
    print(f"Extracting features for {len(word_freq)} words...")

    total_tokens = sum(word_freq.values())

    # Load BERT featurizer
    print("Loading BERT model for seed similarity features...")
    featurizer = SeedSimilarityFeaturizer(
        model_name=config.BERT_MODEL,
        seed_words=seed_words
    )

    words  = list(word_freq.keys())
    rows   = []

    # Basic features
    for i, word in enumerate(words):
        if i % 500 == 0:
            print(f"  Basic features: {i}/{len(words)}")
        rows.append({
            "word":               word,
            "zipf_score":         zipf_frequency(word, "en"),
            "word_length":        len(word),
            "syllable_count":     _syllable_count(word),
            "domain_specificity": _domain_specificity(word, word_freq, total_tokens),
            "wordnet_depth":      _wordnet_depth(word),
            "corpus_freq":        word_freq[word],
        })

    df = pd.DataFrame(rows)

    # BERT seed similarity features (automatic from seed_words.json)
    print("Computing BERT seed similarity features...")
    sim_df = featurizer.transform(words)
    df = pd.concat([df, sim_df], axis=1)

    print(f"Feature extraction complete. Shape: {df.shape}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE MATRIX FOR CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "zipf_score",
    "word_length",
    "syllable_count",
    "domain_specificity",
    "wordnet_depth",
    "sim_layman",
    "sim_student",
    "sim_professional",
    "seed_gap_LP",
    "seed_gap_LS",
]


def get_feature_matrix(df: pd.DataFrame):
    """Return (matrix, columns_used) — only columns present in df."""
    cols = [c for c in FEATURE_COLS if c in df.columns]
    return df[cols].fillna(0).values, cols