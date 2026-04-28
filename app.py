"""
app.py  —  GMM Word Difficulty Analyser
Streamlit dashboard: upload .txt files → run pipeline → visualize results
"""

import os
import io
import json
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="GMM Analysis Workspace",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Syne:wght@400;600;800&display=swap');

/* Root theme */
:root {
    --bg:        #0A0E17;
    --panel:     #111827;
    --border:    #1F2937;
    --accent1:   #4CC9F0;
    --accent2:   #F77F00;
    --accent3:   #E63946;
    --text:      #E2E8F0;
    --subtext:   #64748B;
    --green:     #10B981;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    font-family: 'JetBrains Mono', monospace;
    color: var(--text);
}

[data-testid="stSidebar"] {
    background-color: var(--panel) !important;
    border-right: 1px solid var(--border);
}

/* Header bar */
.top-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 28px;
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    margin-bottom: 24px;
    border-radius: 0 0 12px 12px;
}
.top-header h1 {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 800;
    color: var(--accent1);
    margin: 0;
    letter-spacing: -0.5px;
}
.nav-pills {
    display: flex;
    gap: 8px;
}
.nav-pill {
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    border: 1px solid var(--border);
    color: var(--subtext);
    background: transparent;
}
.nav-pill.active {
    background: var(--accent1);
    color: var(--bg);
    border-color: var(--accent1);
}

/* Section headers */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--subtext);
    margin-bottom: 14px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
}

/* Metric cards */
.metric-row { display: flex; gap: 12px; margin-bottom: 16px; }
.metric-card {
    flex: 1;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
}
.metric-card .label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--subtext);
    margin-bottom: 4px;
}
.metric-card .value {
    font-family: 'Syne', sans-serif;
    font-size: 26px;
    font-weight: 800;
    color: var(--text);
}
.metric-card .delta {
    font-size: 11px;
    color: var(--green);
    margin-top: 2px;
}

/* Cluster badge */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.badge-layman       { background: #4CC9F022; color: #4CC9F0; border: 1px solid #4CC9F044; }
.badge-student      { background: #F77F0022; color: #F77F00; border: 1px solid #F77F0044; }
.badge-professional { background: #E6394622; color: #E63946; border: 1px solid #E6394644; }

/* Upload zone */
.upload-zone {
    border: 2px dashed var(--border);
    border-radius: 12px;
    padding: 32px 20px;
    text-align: center;
    background: #0D1320;
    margin-bottom: 16px;
    transition: border-color 0.2s;
}
.upload-zone:hover { border-color: var(--accent1); }
.upload-icon { font-size: 36px; margin-bottom: 8px; }
.upload-text { font-size: 13px; color: var(--subtext); }

/* Run button */
.stButton > button {
    background: linear-gradient(135deg, var(--accent1), #3A86FF) !important;
    color: #0A0E17 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 13px !important;
    letter-spacing: 1px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 0 !important;
    width: 100% !important;
    text-transform: uppercase !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--panel);
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--subtext);
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    border-radius: 6px;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    background: var(--accent1) !important;
    color: var(--bg) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* Slider */
.stSlider [data-baseweb="slider"] { color: var(--accent1); }

/* Select box */
.stSelectbox [data-baseweb="select"] {
    background: var(--panel) !important;
    border-color: var(--border) !important;
}

/* Progress */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--accent1), #3A86FF) !important;
}

/* Word table rows */
.word-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.word-table th {
    text-align: left; padding: 8px 12px;
    background: #0D1320; color: var(--subtext);
    font-size: 10px; letter-spacing: 1px; text-transform: uppercase;
    border-bottom: 1px solid var(--border);
}
.word-table td {
    padding: 7px 12px;
    border-bottom: 1px solid #1A2030;
    color: var(--text);
}
.word-table tr:hover td { background: #151E2D; }

/* Scrollable word list */
.word-scroll {
    max-height: 320px;
    overflow-y: auto;
    border: 1px solid var(--border);
    border-radius: 8px;
}
.word-scroll::-webkit-scrollbar { width: 4px; }
.word-scroll::-webkit-scrollbar-track { background: var(--panel); }
.word-scroll::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ── Colour constants ──────────────────────────────────────────────────────────
COLORS = {"LAYMAN": "#4CC9F0", "STUDENT": "#F77F00", "PROFESSIONAL": "#E63946"}
LABEL_ORDER = ["LAYMAN", "STUDENT", "PROFESSIONAL"]
DARK_BG  = "#0A0E17"
PANEL_BG = "#111827"
GRID_CLR = "#1F2937"
TEXT_CLR = "#E2E8F0"

plt.rcParams.update({
    "figure.facecolor": DARK_BG, "axes.facecolor": PANEL_BG,
    "axes.edgecolor":   GRID_CLR, "axes.labelcolor": TEXT_CLR,
    "axes.titlecolor":  TEXT_CLR, "xtick.color": "#64748B",
    "ytick.color": "#64748B",     "text.color": TEXT_CLR,
    "grid.color": GRID_CLR,       "grid.linestyle": "--",
    "grid.alpha": 0.4,            "font.family": "monospace",
})


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE (lazy-loaded so Streamlit doesn't crash on import)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_pipeline_modules():
    import config
    from feature_engineering import (
        read_corpus, filter_words, extract_all_features, get_feature_matrix
    )
    from model_training import (
        train_gmm_model, predict_components, predict_probabilities,
        analyze_components, assign_labels_automatic, auto_validate_labels,
        create_vocabulary_datasets
    )
    return (config, read_corpus, filter_words, extract_all_features,
            get_feature_matrix, train_gmm_model, predict_components, predict_probabilities,
            analyze_components, assign_labels_automatic, auto_validate_labels,
            create_vocabulary_datasets)


def run_pipeline(uploaded_files, n_components=3):
    (config, read_corpus, filter_words, extract_all_features,
     get_feature_matrix, train_gmm_model, predict_components, predict_probabilities,
     analyze_components, assign_labels_automatic, auto_validate_labels,
     create_vocabulary_datasets) = load_pipeline_modules()

    seed_file = config.SEED_FILE
    if not os.path.exists(seed_file):
        raise FileNotFoundError(
            "seed_words.json not found. Make sure it exists in your project folder."
        )
    with open(seed_file, encoding="utf-8") as f:
        seeds = json.load(f)

    tmpdir = tempfile.mkdtemp()
    for uf in uploaded_files:
        uf.seek(0)
        text = uf.read().decode("utf-8", errors="ignore")
        with open(os.path.join(tmpdir, uf.name), "w", encoding="utf-8") as fout:
            fout.write(text)

    documents, word_freq = read_corpus(tmpdir)

    total_tokens = sum(word_freq.values())
    if total_tokens < 100:
        raise ValueError(
            "Uploaded file(s) are too small (< 100 tokens). "
            "Please upload real documents with at least a few paragraphs."
        )

    word_freq = filter_words(
        word_freq    = word_freq,
        documents    = documents,
        seed_words   = seeds,
        min_freq     = config.MIN_CORPUS_FREQ,
        min_len      = config.MIN_WORD_LENGTH,
        sigma        = config.ZIPF_SIGMA_FILTER,
        cap_ratio    = config.CAP_RATIO_THRESH,
        min_cap      = config.MIN_CAP_COUNT,
    )

    if len(word_freq) < n_components * 3:
        raise ValueError(
            f"Only {len(word_freq)} words survived filtering — need at least "
            f"{n_components * 3} for {n_components} clusters. "
            "Please upload larger .txt files (ideally 1000+ words each)."
        )

    df = extract_all_features(word_freq, documents, seeds)
    X, feature_cols = get_feature_matrix(df)

    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train GMM
    gmm = train_gmm_model(X_scaled, n_components=n_components)

    # Predict components
    raw_labels = predict_components(gmm, X_scaled)
    probs = predict_probabilities(gmm, X_scaled)

    # Analyze components
    comp_stats = analyze_components(gmm, X_scaled, df, raw_labels)

    # Assign labels automatically
    label_map = assign_labels_automatic(comp_stats, seeds, df, raw_labels, feature_cols)

    # Map integer labels to string labels
    str_labels = np.array([label_map[l] for l in raw_labels])
    labels = str_labels

    # Create datasets
    datasets = create_vocabulary_datasets(df, labels, probs)

    for lbl in list(datasets.keys()):
        d = datasets[lbl]
        if not isinstance(d, pd.DataFrame):
            if isinstance(d, list):
                datasets[lbl] = pd.DataFrame(d) if d and isinstance(d[0], dict) \
                                 else pd.DataFrame({"word": d, "confidence": 1.0})
            else:
                datasets[lbl] = pd.DataFrame(columns=["word", "confidence"])

    return X_scaled, df, labels, probs, comp_stats, label_map, datasets, gmm


# ═══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _legend(label_map):
    seen = []
    for v in label_map.values():
        if v not in seen:
            seen.append(v)
    return [mpatches.Patch(color=COLORS[l], label=l) for l in seen if l in COLORS]


def chart_pca(normalized, labels, label_map):
    pca = PCA(n_components=2, random_state=42)
    r = pca.fit_transform(normalized)
    colors = [COLORS.get(label_map.get(c, ""), "#888") for c in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(r[:, 0], r[:, 1], c=colors, alpha=0.55, s=16, linewidths=0)

    # Draw ellipses per cluster
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    for cid, lbl in label_map.items():
        mask = labels == cid
        if mask.sum() < 5:
            continue
        pts = r[mask]
        mean = pts.mean(axis=0)
        cov  = np.cov(pts.T)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        w, h = 2 * 2.0 * np.sqrt(vals)
        ell = Ellipse(xy=mean, width=w, height=h, angle=angle,
                      color=COLORS.get(lbl, "#888"), alpha=0.12, zorder=0)
        ell_edge = Ellipse(xy=mean, width=w, height=h, angle=angle,
                           fill=False, edgecolor=COLORS.get(lbl, "#888"),
                           linewidth=1.5, alpha=0.6, zorder=1)
        ax.add_patch(ell)
        ax.add_patch(ell_edge)
        ax.plot(*mean, "+", color=COLORS.get(lbl, "#888"), markersize=10, markeredgewidth=2)

    ax.set_title(f"GMM Clustering  (K={len(label_map)})", fontsize=13, pad=10)
    ax.set_xlabel(f"Feature A  —  PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"Feature B  —  PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend(handles=_legend(label_map), fontsize=9, title="Cluster")
    ax.grid(True)
    fig.tight_layout()
    return fig


def chart_distribution(datasets):
    counts = {l: len(datasets[l]) for l in LABEL_ORDER if l in datasets}
    lbls = list(counts.keys())
    vals = list(counts.values())
    clrs = [COLORS[l] for l in lbls]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(lbls, vals, color=clrs, edgecolor=DARK_BG, linewidth=1, width=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + max(vals)*0.01,
                f"{v:,}", ha="center", va="bottom", fontsize=10)
    ax.set_title("Cluster Distribution", fontsize=12)
    ax.set_ylabel("Word Count")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def chart_density(probs, labels, label_map):
    difficulty_labels = [label_map.get(c, "?") for c in labels]
    max_probs = probs.max(axis=1)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for lbl in LABEL_ORDER:
        mask = np.array(difficulty_labels) == lbl
        if mask.sum() < 2:
            continue
        from scipy.stats import gaussian_kde
        data = max_probs[mask]
        xs = np.linspace(0, 1, 200)
        kde = gaussian_kde(data, bw_method=0.15)
        ys = kde(xs)
        ax.fill_between(xs, ys, alpha=0.25, color=COLORS[lbl])
        ax.plot(xs, ys, color=COLORS[lbl], linewidth=2, label=lbl)

    ax.set_title("Density Plots", fontsize=12)
    ax.set_xlabel("Max Posterior Probability")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.grid(True)
    fig.tight_layout()
    return fig


def chart_correlation(df):
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()
    cmap = LinearSegmentedColormap.from_list("rg", ["#E63946", PANEL_BG, "#4CC9F0"])
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, center=0,
                linewidths=0.4, linecolor=DARK_BG, ax=ax,
                annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", fontsize=13, pad=10)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)
    fig.tight_layout()
    return fig


def chart_boxplots(df, labels, label_map):
    plot_df = df.select_dtypes(include=[np.number]).copy()
    plot_df["difficulty"] = [label_map.get(c, "?") for c in labels]
    cols = [c for c in plot_df.columns if c != "difficulty"][:6]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()
    palette = {l: COLORS[l] for l in LABEL_ORDER if l in COLORS}
    for i, col in enumerate(cols):
        order = [l for l in LABEL_ORDER if l in plot_df["difficulty"].unique()]
        sns.boxplot(data=plot_df, x="difficulty", y=col,
                    order=order, palette=palette, ax=axes[i],
                    linewidth=1, fliersize=2)
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel("")
        axes[i].tick_params(axis="x", rotation=15, labelsize=8)
        axes[i].grid(axis="y")
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Feature Distributions by Cluster", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def chart_radar(comp_stats, label_map):
    stat_keys = [k for k in ["avg_zipf_score", "avg_domain_specificity",
                               "avg_word_length", "avg_frequency"]
                 if any(k in v for v in comp_stats.values())]
    if len(stat_keys) < 3:
        return None
    N = len(stat_keys)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"polar": True, "facecolor": PANEL_BG})
    fig.patch.set_facecolor(DARK_BG)
    for cid, stats in comp_stats.items():
        lbl   = label_map.get(cid, f"C{cid}")
        color = COLORS.get(lbl, "#888")
        raw   = [stats.get(k, 0) for k in stat_keys]
        all_v = [[s.get(k, 0) for s in comp_stats.values()] for k in stat_keys]
        normed = [(v - min(av)) / (max(av) - min(av) + 1e-9)
                  for v, av in zip(raw, all_v)]
        vals = normed + normed[:1]
        ax.plot(angles, vals, color=color, linewidth=2, label=lbl)
        ax.fill(angles, vals, color=color, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([k.replace("avg_","").replace("_"," ").title()
                        for k in stat_keys], fontsize=9, color=TEXT_CLR)
    ax.set_yticklabels([])
    ax.set_title("Component Radar", fontsize=12, pad=16, color=TEXT_CLR)
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.grid(color=GRID_CLR, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="top-header">
  <h1>🔬 GMM Analysis Workspace</h1>
  <div class="nav-pills">
    <span class="nav-pill active">Dashboard</span>
    <span class="nav-pill">Models</span>
    <span class="nav-pill">Settings</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT  —  sidebar | main
# ═══════════════════════════════════════════════════════════════════════════════

sidebar, main = st.columns([1, 2.8], gap="large")

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with sidebar:
    st.markdown('<div class="section-title">1. Data & Model Setup</div>', unsafe_allow_html=True)

    # Upload
    st.markdown("""
    <div class="upload-zone">
      <div class="upload-icon">📄</div>
      <div class="upload-text">Drag & Drop .TXT files or browse</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload .txt files",
        type=["txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        st.markdown(f"**{len(uploaded)} file(s) loaded:**")
        for f in uploaded:
            size_kb = len(f.getvalue()) / 1024
            st.markdown(f"&nbsp;&nbsp;📄 `{f.name}` — {size_kb:.1f} KB")

    st.markdown("---")
    st.markdown('<div class="section-title">GMM Configuration</div>', unsafe_allow_html=True)

    k_clusters = st.slider("Number of Clusters (K)", 2, 5, 3)
    cov_type   = st.selectbox("Covariance Type", ["full", "tied", "diag", "spherical"])
    init_method = st.selectbox("Initialization", ["kmeans", "k-means++", "random"])
    max_iter   = st.number_input("Max Iterations", 50, 500, 100, step=50)

    st.markdown("---")
    run_btn = st.button("▶  RUN GMM ANALYSIS")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ═══════════════════════════════════════════════════════════════════════════════

with main:
    st.markdown('<div class="section-title">2. Visualization & Analysis Results</div>',
                unsafe_allow_html=True)

    # ── State ─────────────────────────────────────────────────────────────────
    if "results" not in st.session_state:
        st.session_state.results = None

    # ── Run pipeline ──────────────────────────────────────────────────────────
    if run_btn:
        if not uploaded:
            st.warning("⚠️  Please upload at least one .txt file first.")
        else:
            with st.spinner("Running GMM pipeline..."):
                progress = st.progress(0, text="Loading modules...")
                try:
                    progress.progress(10, "Tokenizing corpus...")
                    results = run_pipeline(uploaded, n_components=k_clusters)
                    progress.progress(100, "Done!")
                    st.session_state.results = results
                    st.success("✅  Analysis complete!")
                except ValueError as e:
                    progress.empty()
                    st.error(f"⚠️  {e}")
                except Exception as e:
                    progress.empty()
                    st.error(f"Unexpected error: {e}")
                    with st.expander("Show full traceback"):
                        st.exception(e)
                finally:
                    progress.empty()

    # ── Display results ───────────────────────────────────────────────────────
    if st.session_state.results is not None:
        normalized, df, labels, probs, comp_stats, label_map, datasets, gmm = st.session_state.results

        difficulty_labels = [label_map.get(c, "?") for c in labels]
        max_probs = probs.max(axis=1)

        # ── Metric cards ──────────────────────────────────────────────────────
        counts = {l: len(datasets.get(l, [])) for l in LABEL_ORDER}
        avg_conf = max_probs.mean()

        # Compute AIC / BIC / Silhouette
        aic = gmm.aic(normalized)
        bic = gmm.bic(normalized)
        try:
            from sklearn.metrics import silhouette_score
            sil = silhouette_score(normalized, labels)
        except Exception:
            sil = 0.0

        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-card">
            <div class="label">Total Words</div>
            <div class="value">{len(df):,}</div>
            <div class="delta">↑ processed</div>
          </div>
          <div class="metric-card">
            <div class="label">AIC</div>
            <div class="value">{aic:,.0f}</div>
          </div>
          <div class="metric-card">
            <div class="label">BIC</div>
            <div class="value">{bic:,.0f}</div>
          </div>
          <div class="metric-card">
            <div class="label">Silhouette</div>
            <div class="value">{sil:.2f}</div>
            <div class="delta">{'↑ good' if sil > 0.4 else '~ moderate'}</div>
          </div>
          <div class="metric-card">
            <div class="label">Avg Confidence</div>
            <div class="value">{avg_conf:.2%}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── PCA scatter (full width) ──────────────────────────────────────────
        st.pyplot(chart_pca(normalized, labels, label_map), use_container_width=True)

        # ── Bottom row: distribution | density | radar ────────────────────────
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="section-title">Cluster Distribution</div>', unsafe_allow_html=True)
            st.pyplot(chart_distribution(datasets), use_container_width=True)
        with c2:
            st.markdown('<div class="section-title">Density Plots</div>', unsafe_allow_html=True)
            try:
                st.pyplot(chart_density(probs, labels, label_map), use_container_width=True)
            except Exception:
                st.info("scipy required for density plots")
        with c3:
            st.markdown('<div class="section-title">Component Radar</div>', unsafe_allow_html=True)
            r = chart_radar(comp_stats, label_map)
            if r:
                st.pyplot(r, use_container_width=True)

        # ── Tabs: deeper analysis ─────────────────────────────────────────────
        st.markdown("---")
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Feature Analysis",
            "📋 Word Explorer",
            "📈 Model Metrics",
            "💾 Export",
        ])

        with tab1:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Feature Correlation**")
                st.pyplot(chart_correlation(df), use_container_width=True)
            with col_b:
                st.markdown("**Feature Box Plots**")
                st.pyplot(chart_boxplots(df, labels, label_map), use_container_width=True)

        with tab2:
            st.markdown("**Browse words by difficulty cluster**")
            selected_label = st.radio(
                "Select cluster",
                LABEL_ORDER,
                horizontal=True,
                label_visibility="collapsed",
            )
            if selected_label in datasets:
                raw_ds = datasets[selected_label]
                if not isinstance(raw_ds, pd.DataFrame):
                    raw_ds = pd.DataFrame({"word": list(raw_ds), "confidence": 1.0})
                ds = raw_ds.copy()

                # Identify word and confidence columns flexibly
                word_col = next((c for c in ["word","Word","token"] if c in ds.columns), ds.columns[0])
                conf_col = next((c for c in ["confidence","conf","score","probability"]
                                 if c in ds.columns), ds.columns[-1] if len(ds.columns) > 1 else word_col)

                search = st.text_input("🔍 Search words", placeholder="Type to filter...")
                if search:
                    ds = ds[ds[word_col].astype(str).str.contains(search, case=False, na=False)]

                display_cols = [word_col] if word_col == conf_col else [word_col, conf_col]
                rename_map   = {word_col: "Word", conf_col: "Confidence"}
                st.dataframe(
                    ds[display_cols].rename(columns=rename_map)
                      .sort_values("Confidence", ascending=False)
                      .reset_index(drop=True),
                    use_container_width=True,
                    height=400,
                )
                st.caption(f"Showing {len(ds):,} of {len(raw_ds):,} words")

        with tab3:
            st.markdown("**Model Evaluation Metrics**")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("AIC", f"{aic:,.1f}")
                st.metric("BIC", f"{bic:,.1f}")
            with m2:
                st.metric("Silhouette Score", f"{sil:.4f}")
                st.metric("Avg Confidence", f"{avg_conf:.4f}")
            with m3:
                st.metric("Components", len(label_map))
                st.metric("Features", normalized.shape[1])

            st.markdown("**Per-Component Statistics**")
            rows = []
            for cid, stats in comp_stats.items():
                lbl = label_map.get(cid, f"C{cid}")
                rows.append({
                    "Cluster": lbl,
                    "Words": counts.get(lbl, 0),
                    "Avg Zipf": f"{stats.get('avg_zipf_score', 0):.3f}",
                    "Avg Domain Spec": f"{stats.get('avg_domain_specificity', 0):.3f}",
                    "Avg Word Length": f"{stats.get('avg_word_length', 0):.2f}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        with tab4:
            st.markdown("**Download vocabulary datasets**")
            for lbl in LABEL_ORDER:
                ds_exp = datasets.get(lbl)
                if ds_exp is not None and len(ds_exp) > 0:
                    if not isinstance(ds_exp, pd.DataFrame):
                        ds_exp = pd.DataFrame({"word": list(ds_exp), "confidence": 1.0})
                    csv = ds_exp.to_csv(index=False)
                    st.download_button(
                        label=f"⬇  Download {lbl.lower()}_vocabulary.csv  ({len(ds_exp):,} words)",
                        data=csv,
                        file_name=f"{lbl.lower()}_vocabulary.csv",
                        mime="text/csv",
                        key=f"dl_{lbl}",
                    )

    else:
        # Placeholder state
        st.markdown("""
        <div style="
            border: 2px dashed #1F2937;
            border-radius: 16px;
            padding: 80px 40px;
            text-align: center;
            background: #0D1320;
        ">
            <div style="font-size: 64px; margin-bottom: 16px;">🔬</div>
            <div style="font-family: 'Syne', sans-serif; font-size: 20px;
                        font-weight: 800; color: #4CC9F0; margin-bottom: 8px;">
                Ready to Analyse
            </div>
            <div style="font-size: 13px; color: #64748B; max-width: 360px; margin: 0 auto;">
                Upload one or more <code>.txt</code> files in the left panel,
                configure your GMM settings, then click <strong>RUN GMM ANALYSIS</strong>.
            </div>
        </div>
        """, unsafe_allow_html=True)