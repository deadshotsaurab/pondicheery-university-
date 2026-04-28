"""
train_gmm_improved.py
Master pipeline for GMM-based Legal Vocabulary Difficulty Classifier.

Pipeline stages:
  1. Read corpus files
  2. Build word frequency map
  3. Auto-filter (proper nouns, zipf bounds, frequency)
  4. Extract features (zipf, syllables, domain TF-IDF, WordNet depth)
  5. Compute BERT seed-similarity features → columns in feature matrix
  6. Normalise with StandardScaler
  7. Semi-supervised GMM training (seed-anchored means + constrained EM)
  8. Auto-assign labels (seed_gap_LP primary signal)
  9. Auto-validate / correct misassigned words
  10. Save datasets + model
  11. Visualise
  12. Evaluate
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config

from feature_engineering import (
    read_corpus,
    filter_words,
    extract_all_features,
    get_feature_matrix,
)
from seed_similarity import SeedSimilarityFeaturizer

from model_training import (
    train_gmm_model,
    predict_components,
    predict_probabilities,
    analyze_components,
    assign_labels_automatic,
    auto_validate_labels,
    create_vocabulary_datasets,
    save_vocabulary_datasets,
    save_model,
    print_final_results,
)

from visualization import create_all_visualizations

# ── Optional evaluate ─────────────────────────────────────────────────────────
try:
    from evaluate_model import evaluate_complete_model
    HAS_EVALUATE = True
except ImportError:
    HAS_EVALUATE = False


# ═══════════════════════════════════════════════════════════════════════════════
# SEED WORDS  (define once here — used throughout pipeline automatically)
# ═══════════════════════════════════════════════════════════════════════════════

SEED_WORDS = {
    "LAYMAN": [
        "house", "car", "food", "water", "money", "people", "child",
        "family", "time", "year", "day", "work", "life", "home",
        "death", "wife", "husband", "son", "daughter", "mother", "father",
        "land", "pay", "give", "take", "buy", "sell", "live",
    ],
    "STUDENT": [
        "analyze", "theory", "evidence", "document", "section",
        "compare", "system", "process", "structure", "function",
        "government", "constitution", "legislation", "provision",
        "defendant", "plaintiff", "appeal", "judgment", "legal",
        "authority", "liability", "obligation", "contract", "clause",
    ],
    "PROFESSIONAL": [
        "habeas", "corpus", "jurisdiction", "tort", "plaintiff",
        "testamentary", "intestate", "promissory", "alienation",
        "subrogation", "estoppel", "mandamus", "certiorari",
        "devolution", "executrix", "probate", "bequest",
        "jurisprudence", "adjudication", "indemnification",
        "sequestration", "interlocutory", "encumbrance", "conveyance",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # ── 1 + 2. Read corpus and word frequency ─────────────────────────────────
    documents, raw_freq = read_corpus(config.DATA_DIR)
    print(f"  Raw vocabulary: {len(raw_freq)} words")

    # ── 3. Auto-filter (proper nouns, zipf bounds) ────────────────────────────
    filtered_freq = filter_words(
        word_freq=raw_freq,
        documents=documents,
        seed_words=SEED_WORDS,
        min_freq=config.MIN_CORPUS_FREQ,
        min_len=config.MIN_WORD_LENGTH,
        sigma=config.ZIPF_SIGMA_FILTER,
        cap_ratio=config.CAP_RATIO_THRESH,
        min_cap=config.MIN_CAP_COUNT,
    )
    print(f"  Filtered vocabulary: {len(filtered_freq)} words")

    # ── 4 + 5. Feature extraction + BERT seed similarities ───────────────────
    df = extract_all_features(
        word_freq=filtered_freq,
        documents=documents,
        seed_words=SEED_WORDS,
    )

    # ── 6. Normalise ──────────────────────────────────────────────────────────
    feature_matrix, feature_cols = get_feature_matrix(df)

    scaler = StandardScaler()
    normalized = scaler.fit_transform(feature_matrix)
    print(f"\nNormalised feature matrix: {normalized.shape}")

    # ── 7. Semi-supervised GMM training ──────────────────────────────────────
    gmm = train_gmm_model(
        normalized,
        n_components=config.N_COMPONENTS if hasattr(config, "N_COMPONENTS") else 3,
    )

    # ── 8. Predictions ────────────────────────────────────────────────────────
    raw_labels = predict_components(gmm, normalized)
    probs      = predict_probabilities(gmm, normalized)

    # ── 9. Analyze + auto-label ───────────────────────────────────────────────
    comp_stats = analyze_components(gmm, normalized, df, raw_labels)
    label_map  = assign_labels_automatic(comp_stats, SEED_WORDS, df,
                                          raw_labels, feature_cols)

    # ── 10. Map labels and skip auto-validate for GMM for now ────────────────
    str_labels = np.array([label_map[l] for l in raw_labels])
    labels = str_labels

    # ── 11. Build + save datasets ─────────────────────────────────────────────
    datasets = create_vocabulary_datasets(df, labels, probs)
    save_vocabulary_datasets(datasets, config.OUTPUT_DIR)
    save_model(gmm, scaler, label_map, feature_cols, config.OUTPUT_DIR)
    print_final_results(gmm, datasets, comp_stats, label_map)

    # ── 12. Visualise ─────────────────────────────────────────────────────────
    create_all_visualizations(
        normalized, df, labels, probs,
        datasets, comp_stats, label_map,
        config.OUTPUT_DIR,
    )

    # ── 13. Evaluate (optional) ───────────────────────────────────────────────
    if HAS_EVALUATE:
        try:
            report = evaluate_complete_model(
                normalized, df, labels, probs, label_map, gmm, config.OUTPUT_DIR
            )
        except Exception as e:
            print(f"  Evaluation skipped: {e}")
    else:
        print("\n  (evaluate_model.py not found — skipping evaluation)")

    print("\n✔  Pipeline complete.")
    return datasets, label_map, gmm


if __name__ == "__main__":
    main()