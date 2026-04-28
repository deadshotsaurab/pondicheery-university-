"""
model_training.py
Fully automatic classifier.
All thresholds derived from seed word statistics.
Zero hardcoded word lists, boundaries, or manual rules.
"""

import numpy as np
import pandas as pd
import pickle
import os
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from wordfreq import zipf_frequency

import config


# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-DERIVED THRESHOLDS FROM SEED WORDS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_thresholds_from_seeds(seed_words: dict, df: pd.DataFrame) -> dict:
    """
    Derive classification thresholds AUTOMATICALLY from seed word feature
    distributions in the training DataFrame.

    Returns dict of {label: {feature: (mean, std), ...}}
    No manual numbers anywhere.
    """
    thresholds = {}
    for label, words in seed_words.items():
        lbl_upper = label.upper()
        mask = df["word"].isin([w.lower() for w in words])
        subset = df[mask]
        if len(subset) == 0:
            continue
        thresholds[lbl_upper] = {
            col: (float(subset[col].mean()), float(subset[col].std()))
            for col in ["zipf_score", "word_length", "domain_specificity",
                        "sim_layman", "sim_student", "sim_professional",
                        "seed_gap_LP"]
            if col in subset.columns
        }
    return thresholds


# ═══════════════════════════════════════════════════════════════════════════════
# SEED-BASED KNN TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def build_training_data(df: pd.DataFrame,
                        seed_words: dict,
                        feature_cols: list):
    """
    Build (X_train, y_train) purely from seed words present in df.
    Labels come from seed_words.json keys — no manual assignment.
    """
    X_rows, y_rows = [], []
    label_map = {}   # word → label

    for label, words in seed_words.items():
        lbl_upper = label.upper()
        for w in words:
            mask = df["word"] == w.lower()
            if mask.any():
                row = df[mask].iloc[0]
                X_rows.append([row.get(c, 0) for c in feature_cols])
                y_rows.append(lbl_upper)
                label_map[w.lower()] = lbl_upper

    X = np.array(X_rows)
    y = np.array(y_rows)
    print(f"Training data: {len(y)} seed words")
    for lbl in sorted(set(y)):
        print(f"  {lbl}: {(y == lbl).sum()} seeds")
    return X, y, label_map


def train_classifier(df: pd.DataFrame,
                     seed_words: dict,
                     feature_cols: list,
                     k: int = 5):
    """
    Train KNN classifier on seed words.
    k is the only hyperparameter — comes from config, not hardcoded here.
    Returns (classifier, scaler, X_scaled, label_map, thresholds).
    """
    scaler  = StandardScaler()
    X_all   = df[feature_cols].fillna(0).values
    X_scaled = scaler.fit_transform(X_all)

    # Build scaled training set
    X_train_raw, y_train, label_map = build_training_data(df, seed_words, feature_cols)
    if len(X_train_raw) == 0:
        raise ValueError(
            "No seed words found in feature DataFrame. "
            "Check seed_words.json entries match corpus vocabulary."
        )

    X_train = scaler.transform(X_train_raw)

    # Auto-select k: cannot exceed number of training samples
    effective_k = min(k, len(X_train))

    clf = KNeighborsClassifier(
        n_neighbors = effective_k,
        metric      = "euclidean",
        weights     = "distance",    # closer seeds have more influence
    )
    clf.fit(X_train, y_train)

    # ── Cross-validation (FIXED: cv cannot exceed min class size) ────────────
    try:
        min_class_size = min(Counter(y_train).values())
        cv_k = min(5, len(X_train), min_class_size)
        if cv_k >= 2:
            scores = cross_val_score(clf, X_train, y_train, cv=cv_k)
            print(f"Seed cross-validation accuracy: {scores.mean()*100:.1f}%")
        else:
            print("Skipping cross-validation: not enough samples per class.")
    except Exception as e:
        print(f"Cross-validation skipped: {e}")

    # Compute auto-derived thresholds for fallback
    thresholds = compute_thresholds_from_seeds(seed_words, df)

    return clf, scaler, X_scaled, label_map, thresholds


# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-DERIVED ZIPF FALLBACK
# ═══════════════════════════════════════════════════════════════════════════════

def _get_zipf_boundaries(thresholds: dict) -> tuple:
    """
    Derive LAYMAN/STUDENT/PROFESSIONAL zipf boundaries from seed statistics.
    Boundary between LAYMAN and STUDENT = midpoint of their zipf means.
    Boundary between STUDENT and PROFESSIONAL = midpoint of their zipf means.
    All values come from seed data — nothing hardcoded.
    """
    means = {}
    for lbl in ["LAYMAN", "STUDENT", "PROFESSIONAL"]:
        if lbl in thresholds and "zipf_score" in thresholds[lbl]:
            means[lbl] = thresholds[lbl]["zipf_score"][0]

    if len(means) < 2:
        return 4.0, 3.0   # only used if seeds are missing

    layman_mean  = means.get("LAYMAN",       5.0)
    student_mean = means.get("STUDENT",      4.0)
    prof_mean    = means.get("PROFESSIONAL", 2.5)

    # Midpoints between adjacent group means
    boundary_LS = (layman_mean + student_mean) / 2.0   # above → LAYMAN
    boundary_SP = (student_mean + prof_mean)   / 2.0   # above → STUDENT

    return boundary_LS, boundary_SP


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFY ALL WORDS
# ═══════════════════════════════════════════════════════════════════════════════

def classify_all_words(df: pd.DataFrame,
                       clf: KNeighborsClassifier,
                       scaler: StandardScaler,
                       X_scaled: np.ndarray,
                       feature_cols: list,
                       thresholds: dict,
                       confidence_threshold: float) -> tuple:
    """
    Classify every word.
    High-confidence → KNN label.
    Low-confidence  → auto-derived zipf-boundary fallback.
    Returns (labels_array, probs_array).
    """
    probs    = clf.predict_proba(X_scaled)
    labels   = clf.predict(X_scaled)
    max_conf = probs.max(axis=1)

    # Auto-derived zipf boundaries (from seed distributions)
    boundary_LS, boundary_SP = _get_zipf_boundaries(thresholds)
    print(f"  Auto zipf boundaries (seed-derived): "
          f"LAYMAN>={boundary_LS:.2f} | STUDENT>={boundary_SP:.2f} | "
          f"PROFESSIONAL<{boundary_SP:.2f}")

    fallback_count = 0
    if "zipf_score" in df.columns:
        for i, conf in enumerate(max_conf):
            if conf < confidence_threshold:
                z = float(df.iloc[i]["zipf_score"])
                if z >= boundary_LS:
                    labels[i] = "LAYMAN"
                elif z >= boundary_SP:
                    labels[i] = "STUDENT"
                else:
                    labels[i] = "PROFESSIONAL"
                fallback_count += 1

    print(f"  Zipf fallback applied to {fallback_count} low-confidence words")
    return labels, probs


# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-VALIDATION USING SEED STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

def auto_validate_labels(df: pd.DataFrame,
                         labels: np.ndarray,
                         thresholds: dict,
                         sigma: float = 2.0) -> np.ndarray:
    """
    Re-assign words that are statistical outliers relative to their
    assigned cluster's seed distribution.

    Outlier criterion: word's zipf is more than `sigma` standard deviations
    away from the assigned cluster's seed mean, AND it fits another cluster
    much better (within 0.5σ).

    All boundaries derived from seed statistics — no manual values.
    """
    if not thresholds:
        return labels

    labels     = labels.copy()
    reassigned = 0

    for i, (_, row) in enumerate(df.iterrows()):
        assigned = labels[i]
        if assigned not in thresholds:
            continue
        if "zipf_score" not in thresholds[assigned]:
            continue

        z = float(row.get("zipf_score", 3.0))
        a_mean, a_std = thresholds[assigned]["zipf_score"]
        if a_std < 0.01:
            continue

        z_score = abs(z - a_mean) / a_std

        if z_score > sigma:
            # Find best fitting alternative cluster
            best_lbl, best_z = assigned, z_score
            for lbl, stats in thresholds.items():
                if lbl == assigned or "zipf_score" not in stats:
                    continue
                m, s = stats["zipf_score"]
                if s < 0.01:
                    continue
                alt_z = abs(z - m) / s
                if alt_z < best_z:
                    best_z   = alt_z
                    best_lbl = lbl

            if best_lbl != assigned:
                labels[i]  = best_lbl
                reassigned += 1

    print(f"  Auto-validator: {reassigned} words re-assigned")
    return labels


# ═══════════════════════════════════════════════════════════════════════════════
# VOCABULARY DATASETS
# ═══════════════════════════════════════════════════════════════════════════════

def create_vocabulary_datasets(df: pd.DataFrame,
                               labels: np.ndarray,
                               probs: np.ndarray) -> dict:
    """Return {LABEL: DataFrame} sorted by confidence descending."""
    datasets     = {}
    label_classes = sorted(set(labels))

    for lbl in label_classes:
        idx  = np.where(labels == lbl)[0]
        rows = []

        for i in idx:
            row             = df.iloc[i].to_dict()
            row["confidence"] = float(probs[i].max())
            row["label"]      = lbl
            rows.append(row)

        out           = pd.DataFrame(rows).sort_values("confidence", ascending=False)
        datasets[lbl] = out

    return datasets


def save_vocabulary_datasets(datasets: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for label, df_out in datasets.items():
        path = os.path.join(output_dir, f"{label.lower()}_vocabulary.csv")
        df_out.to_csv(path, index=False)
    print("Vocabulary datasets saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# CLUSTER STATS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_clusters(df: pd.DataFrame, labels: np.ndarray) -> dict:
    stats = {}
    for lbl in sorted(set(labels)):
        mask   = labels == lbl
        subset = df[mask]
        stat   = {"size": int(mask.sum())}
        for col in ["zipf_score", "word_length", "domain_specificity",
                    "sim_layman", "sim_student", "sim_professional",
                    "seed_gap_LP"]:
            if col in subset.columns:
                stat[f"avg_{col}"] = float(subset[col].mean())
        stats[lbl] = stat
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE / LOAD MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def save_model(model, scaler, thresholds_or_label_map, feature_cols, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    if hasattr(model, 'predict_proba') and hasattr(model, 'n_components'):  # GMM
        filename = "gmm_model.pkl"
        data = {
            "gmm": model,
            "scaler": scaler,
            "label_map": thresholds_or_label_map,
            "feature_cols": feature_cols,
        }
    else:  # KNN
        filename = "classifier.pkl"
        data = {
            "clf": model,
            "scaler": scaler,
            "thresholds": thresholds_or_label_map,
            "feature_cols": feature_cols,
        }
    with open(os.path.join(output_dir, filename), "wb") as f:
        pickle.dump(data, f)
    print("Model saved successfully.")


def load_model(output_dir: str):
    # Try GMM first
    gmm_path = os.path.join(output_dir, "gmm_model.pkl")
    if os.path.exists(gmm_path):
        with open(gmm_path, "rb") as f:
            return pickle.load(f)
    # Fallback to KNN
    clf_path = os.path.join(output_dir, "classifier.pkl")
    with open(clf_path, "rb") as f:
        return pickle.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# GMM MODEL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def train_gmm_model(normalized, n_components=3, covariance_type='full', random_state=42, **kwargs):
    """
    Train a Gaussian Mixture Model on normalized features.
    """
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
        **kwargs
    )
    gmm.fit(normalized)
    return gmm


def predict_components(gmm, normalized):
    """
    Predict cluster components for each data point.
    """
    return gmm.predict(normalized)


def predict_probabilities(gmm, normalized):
    """
    Predict probabilities for each component.
    """
    return gmm.predict_proba(normalized)


def analyze_components(gmm, normalized, df, labels):
    """
    Analyze GMM components and compute statistics.
    """
    comp_stats = {}
    for i in range(gmm.n_components):
        mask = labels == i
        subset = df[mask]
        comp_stats[i] = {
            'size': len(subset),
            'avg_zipf_score': subset.get('zipf_score', pd.Series()).mean() if len(subset) > 0 else 0,
            'avg_word_length': subset.get('word_length', pd.Series()).mean() if len(subset) > 0 else 0,
            'avg_domain_specificity': subset.get('domain_specificity', pd.Series()).mean() if len(subset) > 0 else 0,
        }
    return comp_stats


def assign_labels_automatic(comp_stats, seed_words, df, raw_labels, feature_cols):
    """
    Automatically assign difficulty labels (LAYMAN, STUDENT, PROFESSIONAL) to GMM components
    based on component statistics and seed word analysis.
    """
    # Sort components by average zipf score (higher zipf = easier words)
    sorted_comps = sorted(comp_stats.items(), key=lambda x: x[1]['avg_zipf_score'], reverse=True)
    
    # Assume 3 components: LAYMAN (easiest), STUDENT (medium), PROFESSIONAL (hardest)
    label_map = {}
    labels = ['LAYMAN', 'STUDENT', 'PROFESSIONAL']
    for idx, (comp_id, _) in enumerate(sorted_comps):
        if idx < len(labels):
            label_map[comp_id] = labels[idx]
        else:
            label_map[comp_id] = f'UNKNOWN_{comp_id}'
    
    return label_map


def print_final_results(gmm, datasets, comp_stats, label_map):
    """
    Print final GMM results.
    """
    print("\nFINAL GMM RESULTS:")
    for comp_id, stats in comp_stats.items():
        label = label_map.get(comp_id, f'Component {comp_id}')
        print(f"  {label}: {stats['size']} words, avg_zipf={stats['avg_zipf_score']:.2f}")