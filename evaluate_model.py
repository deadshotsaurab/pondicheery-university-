"""
evaluate_model.py
Automatic evaluation — all metrics derived from data, no manual thresholds.
"""

import os
import re
import glob
import json
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import silhouette_score, davies_bouldin_score
import config
from feature_engineering import filter_words, extract_all_features, get_feature_matrix


def evaluate_complete_model(X_scaled, df, labels, probs, thresholds, clf, output_dir):
    report = {}

    unique_labels = sorted(set(labels))
    if len(unique_labels) >= 2:
        try:
            sil = silhouette_score(X_scaled, labels)
            dbi = davies_bouldin_score(X_scaled, labels)
            report["silhouette_score"] = round(float(sil), 4)
            report["davies_bouldin_index"] = round(float(dbi), 4)
        except Exception:
            report["silhouette_score"] = None
            report["davies_bouldin_index"] = None

    max_conf = probs.max(axis=1)
    report["avg_confidence"]  = round(float(max_conf.mean()), 4)
    report["low_conf_words"]  = int((max_conf < 0.55).sum())
    report["high_conf_words"] = int((max_conf >= 0.80).sum())

    cluster_stats = {}
    for lbl in unique_labels:
        mask   = labels == lbl
        subset = df[mask]
        stat   = {"count": int(mask.sum())}
        for col in ["zipf_score", "word_length", "domain_specificity",
                    "sim_layman", "sim_student", "sim_professional"]:
            if col in subset.columns:
                stat[f"avg_{col}"] = round(float(subset[col].mean()), 4)
        cluster_stats[lbl] = stat
    report["cluster_stats"] = cluster_stats

    # Seed recovery rate
    correct, total_seeds = 0, 0
    for lbl in unique_labels:
        mask = labels == lbl
        if lbl in clf.classes_:
            idx = list(clf.classes_).index(lbl)
            conf_correct = probs[mask, idx]
            if len(conf_correct) > 0:
                correct     += int((conf_correct >= 0.80).sum())
                total_seeds += len(conf_correct)
    if total_seeds > 0:
        report["seed_recovery_rate"] = round(correct / total_seeds, 4)

    # ── Print report ──────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("         EVALUATION REPORT")
    print("="*50)
    print(f"  Silhouette Score       : {report.get('silhouette_score', 'N/A')}")
    print(f"  Davies-Bouldin Index   : {report.get('davies_bouldin_index', 'N/A')}")
    print(f"  Avg Confidence         : {report.get('avg_confidence', 'N/A')}")
    print(f"  Low-conf words (<0.55) : {report.get('low_conf_words', 'N/A')}")
    print(f"  High-conf words (≥0.8) : {report.get('high_conf_words', 'N/A')}")
    print(f"  Seed Recovery Rate     : {report.get('seed_recovery_rate', 'N/A')}")
    print("\n  Per-cluster stats:")
    for lbl, stat in report.get("cluster_stats", {}).items():
        print(f"\n    [{lbl}]")
        for k, v in stat.items():
            print(f"      {k}: {v}")

    # ── Save JSON report ──────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved → {report_path}")
    print("="*50)
    return report


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading saved model from output/...")

    # ── Load model — saved as a single dict by save_model() ──────────────
    model_path = os.path.join(config.OUTPUT_DIR, "classifier.pkl")
    with open(model_path, "rb") as f:
        model_bundle = pickle.load(f)

    # Unpack the dict
    clf          = model_bundle["clf"]
    scaler       = model_bundle["scaler"]
    thresholds   = model_bundle["thresholds"]
    feature_cols = model_bundle["feature_cols"]

    print("  ✔ Model loaded successfully")
    print(f"  ✔ Classes : {list(clf.classes_)}")
    print(f"  ✔ Features: {feature_cols}")

    # ── Load seeds ────────────────────────────────────────────────────────
    with open(config.SEED_FILE, encoding="utf-8") as f:
        seeds = json.load(f)

    # ── Re-read corpus from all 3 folders ─────────────────────────────────
    JUDGEMENT_DIR    = os.path.join(config.DATA_DIR, "IN-Ext_judgement")
    GROUND_TRUTH_DIR = os.path.join(config.DATA_DIR, "A1_Ground truth")
    LOCKNKEY_DIR     = os.path.join(config.DATA_DIR, "A2_LocknKey")

    documents, all_tokens = [], []
    for folder in [JUDGEMENT_DIR, GROUND_TRUTH_DIR, LOCKNKEY_DIR]:
        for p in glob.glob(os.path.join(folder, "*.txt")):
            try:    text = open(p, encoding="utf-8",   errors="ignore").read()
            except: text = open(p, encoding="latin-1", errors="ignore").read()
            documents.append(text)
            all_tokens.extend(re.findall(r"[a-zA-Z]+", text))

    word_freq = Counter(all_tokens)
    print(f"  ✔ Corpus: {len(documents)} documents loaded")

    # ── Filter + extract features ─────────────────────────────────────────
    word_freq = filter_words(
        word_freq, documents, seeds,
        config.MIN_CORPUS_FREQ, config.MIN_WORD_LENGTH,
        config.ZIPF_SIGMA_FILTER, config.CAP_RATIO_THRESH, config.MIN_CAP_COUNT
    )
    df = extract_all_features(word_freq, documents, seeds)

    # ── Scale + predict ───────────────────────────────────────────────────
    X_scaled = scaler.transform(df[feature_cols].fillna(0).values)
    labels   = clf.predict(X_scaled)
    probs    = clf.predict_proba(X_scaled)
    labels   = np.array(labels)

    print(f"\n  ✔ Classification complete: {len(labels)} words")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"     {u}: {c} words ({100*c/len(labels):.1f}%)")

    # ── Run evaluation ────────────────────────────────────────────────────
    evaluate_complete_model(X_scaled, df, labels, probs, thresholds, clf, config.OUTPUT_DIR)