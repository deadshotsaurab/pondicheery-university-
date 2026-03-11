"""
visualization.py
All visualizations. Accepts the full pipeline output.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import Counter

# ── Palette: one colour per label (auto-assigned by sorted order) ──────────
_PALETTE = {
    "LAYMAN":       "#4C9BE8",
    "STUDENT":      "#F5A623",
    "PROFESSIONAL": "#E8524C",
}

def _colour(label: str) -> str:
    return _PALETTE.get(label, "#888888")


def _confidence_ellipse(ax, x, y, color, n_std=2.0, **kwargs):
    """Draw a covariance ellipse around a cluster."""
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    if cov.shape != (2, 2):
        return
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    w, h  = 2 * n_std * np.sqrt(eigenvalues)
    ell   = Ellipse(xy=(np.mean(x), np.mean(y)), width=w, height=h,
                    angle=angle, edgecolor=color, fc="none",
                    lw=2, linestyle="--", **kwargs)
    ax.add_patch(ell)


def create_all_visualizations(features, df=None, labels=None, probs=None,
                               datasets=None, comp_stats=None,
                               label_map=None, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating visualizations → {output_dir}")

    unique_labels = sorted(set(labels)) if labels is not None else []
    colours = [_colour(l) for l in unique_labels]
    colour_map = {l: _colour(l) for l in unique_labels}

    # ── numeric label array for plots ────────────────────────────────────
    lbl2int = {l: i for i, l in enumerate(unique_labels)}
    int_labels = np.array([lbl2int[l] for l in labels]) if labels is not None else None

    # ── 01 PCA SCATTER ────────────────────────────────────────────────────
    try:
        pca    = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(features)
        fig, ax = plt.subplots(figsize=(9, 7))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")
        for lbl in unique_labels:
            mask = labels == lbl
            x, y = coords[mask, 0], coords[mask, 1]
            ax.scatter(x, y, c=colour_map[lbl], label=lbl,
                       alpha=0.65, s=18, linewidths=0)
            _confidence_ellipse(ax, x, y, colour_map[lbl])
            if len(x):
                ax.scatter(x.mean(), y.mean(), c=colour_map[lbl],
                           marker="*", s=250, zorder=5, edgecolors="white", lw=0.8)
        ax.set_title("PCA — Word Clusters", color="white", fontsize=14, pad=12)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", color="#aaa")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", color="#aaa")
        ax.tick_params(colors="#aaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.legend(facecolor="#0f3460", edgecolor="#333", labelcolor="white")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "01_pca_clusters.png"), dpi=150)
        plt.close()
        print("  ✔  01_pca_clusters.png")
    except Exception as e:
        print(f"  ✗  01_pca_clusters.png  ({e})")

    # ── 02 t-SNE SCATTER ──────────────────────────────────────────────────
    try:
        perp   = min(30, max(5, len(features) // 10))
        tsne   = TSNE(n_components=2, random_state=42, perplexity=perp,
                      n_iter=500, verbose=0)
        coords2 = tsne.fit_transform(features)
        fig, ax = plt.subplots(figsize=(9, 7))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")
        for lbl in unique_labels:
            mask = labels == lbl
            ax.scatter(coords2[mask, 0], coords2[mask, 1],
                       c=colour_map[lbl], label=lbl,
                       alpha=0.65, s=18, linewidths=0)
        ax.set_title("t-SNE — Word Clusters", color="white", fontsize=14, pad=12)
        ax.tick_params(colors="#aaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.legend(facecolor="#0f3460", edgecolor="#333", labelcolor="white")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "02_tsne_clusters.png"), dpi=150)
        plt.close()
        print("  ✔  02_tsne_clusters.png")
    except Exception as e:
        print(f"  ✗  02_tsne_clusters.png  ({e})")

    # ── 03 CLUSTER DISTRIBUTION ───────────────────────────────────────────
    try:
        counts = {l: int((labels == l).sum()) for l in unique_labels}
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")
        bars = ax.bar(list(counts.keys()), list(counts.values()),
                      color=[colour_map[l] for l in counts],
                      width=0.5, edgecolor="#333", linewidth=0.8)
        for bar, val in zip(bars, counts.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    str(val), ha="center", color="white", fontsize=11)
        ax.set_title("Word Count per Difficulty Level",
                     color="white", fontsize=13, pad=12)
        ax.set_ylabel("Count", color="#aaa")
        ax.tick_params(colors="#aaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "03_cluster_distribution.png"), dpi=150)
        plt.close()
        print("  ✔  03_cluster_distribution.png")
    except Exception as e:
        print(f"  ✗  03_cluster_distribution.png  ({e})")

    # ── 04 FEATURE CORRELATION HEATMAP ────────────────────────────────────
    try:
        if df is not None:
            num_cols = [c for c in df.select_dtypes(include=np.number).columns
                        if c not in ["corpus_freq"]]
            corr = df[num_cols].corr()
            fig, ax = plt.subplots(figsize=(11, 9))
            fig.patch.set_facecolor("#1a1a2e")
            ax.set_facecolor("#16213e")
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                        linewidths=0.4, linecolor="#333",
                        annot_kws={"size": 8}, ax=ax,
                        cbar_kws={"shrink": 0.8})
            ax.set_title("Feature Correlation", color="white", fontsize=13, pad=12)
            ax.tick_params(colors="#ccc", labelsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "04_feature_correlation.png"), dpi=150)
            plt.close()
            print("  ✔  04_feature_correlation.png")
    except Exception as e:
        print(f"  ✗  04_feature_correlation.png  ({e})")

    # ── 05 FEATURE BOX PLOTS ──────────────────────────────────────────────
    try:
        if df is not None and labels is not None:
            plot_cols = [c for c in ["zipf_score", "word_length",
                                     "domain_specificity", "seed_gap_LP"]
                         if c in df.columns]
            df_plot = df[plot_cols].copy()
            df_plot["label"] = labels
            fig, axes = plt.subplots(1, len(plot_cols), figsize=(4*len(plot_cols), 5))
            fig.patch.set_facecolor("#1a1a2e")
            if len(plot_cols) == 1:
                axes = [axes]
            for ax, col in zip(axes, plot_cols):
                ax.set_facecolor("#16213e")
                for lbl in unique_labels:
                    vals = df_plot[df_plot["label"] == lbl][col].dropna()
                    ax.boxplot(vals, positions=[lbl2int[lbl]],
                               patch_artist=True,
                               boxprops=dict(facecolor=colour_map[lbl], alpha=0.7),
                               medianprops=dict(color="white"),
                               whiskerprops=dict(color="#aaa"),
                               capprops=dict(color="#aaa"),
                               flierprops=dict(marker="o", color="#555", ms=2))
                ax.set_title(col, color="white", fontsize=10)
                ax.set_xticks(range(len(unique_labels)))
                ax.set_xticklabels(unique_labels, rotation=15, color="#aaa", fontsize=8)
                ax.tick_params(colors="#aaa")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#333")
            plt.suptitle("Feature Distribution by Cluster",
                         color="white", fontsize=12, y=1.01)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "05_feature_boxplots.png"),
                        dpi=150, bbox_inches="tight")
            plt.close()
            print("  ✔  05_feature_boxplots.png")
    except Exception as e:
        print(f"  ✗  05_feature_boxplots.png  ({e})")

    # ── 06 CONFIDENCE HISTOGRAM ───────────────────────────────────────────
    try:
        if probs is not None:
            max_conf = probs.max(axis=1)
            fig, ax  = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor("#1a1a2e")
            ax.set_facecolor("#16213e")
            for lbl in unique_labels:
                mask = labels == lbl
                ax.hist(max_conf[mask], bins=30, alpha=0.65,
                        color=colour_map[lbl], label=lbl, edgecolor="#333")
            ax.axvline(0.55, color="white", linestyle="--", linewidth=1.2,
                       label="Confidence threshold")
            ax.set_title("Classification Confidence", color="white", fontsize=13)
            ax.set_xlabel("Max posterior probability", color="#aaa")
            ax.set_ylabel("Count", color="#aaa")
            ax.tick_params(colors="#aaa")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333")
            ax.legend(facecolor="#0f3460", edgecolor="#333", labelcolor="white")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "06_confidence_histogram.png"), dpi=150)
            plt.close()
            print("  ✔  06_confidence_histogram.png")
    except Exception as e:
        print(f"  ✗  06_confidence_histogram.png  ({e})")

    # ── 07 TOP WORDS PER CLUSTER ──────────────────────────────────────────
    try:
        if datasets is not None:
            fig, axes = plt.subplots(1, len(datasets),
                                     figsize=(6*len(datasets), 6))
            fig.patch.set_facecolor("#1a1a2e")
            if len(datasets) == 1:
                axes = [axes]
            for ax, (lbl, dset) in zip(axes, sorted(datasets.items())):
                ax.set_facecolor("#16213e")
                top = dset.head(20)
                words = top["word"].tolist() if "word" in top.columns else []
                confs = top["confidence"].tolist() if "confidence" in top.columns \
                        else [1.0] * len(words)
                ax.barh(range(len(words)), confs,
                        color=colour_map.get(lbl, "#888"), alpha=0.8,
                        edgecolor="#333")
                ax.set_yticks(range(len(words)))
                ax.set_yticklabels(words, color="white", fontsize=9)
                ax.invert_yaxis()
                ax.set_title(f"{lbl}\nTop 20 words", color="white", fontsize=11)
                ax.set_xlabel("Confidence", color="#aaa")
                ax.tick_params(colors="#aaa")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#333")
            plt.suptitle("Top Words per Difficulty Level",
                         color="white", fontsize=13, y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "07_top_words.png"),
                        dpi=150, bbox_inches="tight")
            plt.close()
            print("  ✔  07_top_words.png")
    except Exception as e:
        print(f"  ✗  07_top_words.png  ({e})")

    # ── 08 ZIPF DISTRIBUTION PER CLUSTER ─────────────────────────────────
    try:
        if df is not None and labels is not None and "zipf_score" in df.columns:
            fig, ax = plt.subplots(figsize=(9, 5))
            fig.patch.set_facecolor("#1a1a2e")
            ax.set_facecolor("#16213e")
            for lbl in unique_labels:
                mask = labels == lbl
                vals = df[mask]["zipf_score"].dropna()
                ax.hist(vals, bins=30, alpha=0.6, color=colour_map[lbl],
                        label=lbl, edgecolor="#333", density=True)
            ax.set_title("Zipf Score Distribution (word frequency)",
                         color="white", fontsize=13)
            ax.set_xlabel("Zipf score  (higher = more common word)",
                          color="#aaa")
            ax.set_ylabel("Density", color="#aaa")
            ax.tick_params(colors="#aaa")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333")
            ax.legend(facecolor="#0f3460", edgecolor="#333", labelcolor="white")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "08_zipf_distribution.png"), dpi=150)
            plt.close()
            print("  ✔  08_zipf_distribution.png")
    except Exception as e:
        print(f"  ✗  08_zipf_distribution.png  ({e})")

    print("All visualizations generated successfully.")