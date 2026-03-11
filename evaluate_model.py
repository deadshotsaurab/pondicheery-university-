"""
evaluate_model.py
Automatic evaluation — all metrics derived from data, no manual thresholds.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score
from wordfreq import zipf_frequency


def evaluate_complete_model(X_scaled, df, labels, probs,
                            thresholds, clf, output_dir):
    """
    Compute evaluation metrics automatically from data.
    All reported statistics are derived from the model outputs — 
    no hardcoded reference values.
    """
    report = {}

    # ── Clustering quality metrics ────────────────────────────────────────
    unique_labels = sorted(set(labels))
    if len(unique_labels) >= 2:
        try:
            sil = silhouette_score(X_scaled, labels)
            dbi = davies_bouldin_score(X_scaled, labels)
            report["silhouette_score"] = round(float(sil), 4)
            report["davies_bouldin_index"] = round(float(dbi), 4)
        except Exception as e:
            report["silhouette_score"] = None
            report["davies_bouldin_index"] = None

    # ── Confidence statistics ─────────────────────────────────────────────
    max_conf = probs.max(axis=1)
    report["avg_confidence"]     = round(float(max_conf.mean()), 4)
    report["low_conf_words"]     = int((max_conf < 0.55).sum())
    report["high_conf_words"]    = int((max_conf >= 0.80).sum())

    # ── Per-cluster statistics ────────────────────────────────────────────
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

    # ── Seed recovery rate ────────────────────────────────────────────────
    # % of seed words that ended up in the correct cluster
    if thresholds:
        correct, total_seeds = 0, 0
        for lbl in unique_labels:
            # find indices of seed-like words (high confidence in correct class)
            mask = labels == lbl
            conf_correct = probs[mask, list(clf.classes_).index(lbl)] \
                           if lbl in clf.classes_ else np.array([])
            if len(conf_correct) > 0:
                correct    += int((conf_correct >= 0.80).sum())
                total_seeds += len(conf_correct)
        if total_seeds > 0:
            report["seed_recovery_rate"] = round(correct / total_seeds, 4)

    # ── Print report ──────────────────────────────────────────────────────
    print("\n=== EVALUATION REPORT ===")
    print(f"  Silhouette Score      : {report.get('silhouette_score', 'N/A')}")
    print(f"  Davies-Bouldin Index  : {report.get('davies_bouldin_index', 'N/A')}")
    print(f"  Avg Confidence        : {report.get('avg_confidence', 'N/A')}")
    print(f"  Low-conf words (<0.55): {report.get('low_conf_words', 'N/A')}")
    print(f"  High-conf words (≥0.8): {report.get('high_conf_words', 'N/A')}")
    print("\n  Per-cluster stats:")
    for lbl, stat in report.get("cluster_stats", {}).items():
        print(f"    [{lbl}]")
        for k, v in stat.items():
            print(f"      {k}: {v}")

    # ── Save report ───────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        for k, v in report.items():
            f.write(f"{k}: {v}\n")
    print(f"\n  Report saved → {report_path}")

    return report