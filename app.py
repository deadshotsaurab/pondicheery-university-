"""
app.py  —  GMM Word Difficulty Analyser
Streamlit dashboard: upload .txt files → run pipeline → visualize results
ENHANCED: Document Readability Analyzer section added
"""

import os
import io
import re
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
.top-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 28px; background: var(--panel);
    border-bottom: 1px solid var(--border);
    margin-bottom: 24px; border-radius: 0 0 12px 12px;
}
.top-header h1 {
    font-family: 'Syne', sans-serif; font-size: 22px; font-weight: 800;
    color: var(--accent1); margin: 0; letter-spacing: -0.5px;
}
.nav-pills { display: flex; gap: 8px; }
.nav-pill {
    padding: 6px 14px; border-radius: 20px; font-size: 12px;
    font-weight: 600; cursor: pointer; border: 1px solid var(--border);
    color: var(--subtext); background: transparent;
}
.nav-pill.active { background: var(--accent1); color: var(--bg); border-color: var(--accent1); }
.section-title {
    font-family: 'Syne', sans-serif; font-size: 11px; font-weight: 700;
    letter-spacing: 2px; text-transform: uppercase; color: var(--subtext);
    margin-bottom: 14px; padding-bottom: 6px; border-bottom: 1px solid var(--border);
}
.metric-row { display: flex; gap: 12px; margin-bottom: 16px; }
.metric-card {
    flex: 1; background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 16px;
}
.metric-card .label {
    font-size: 10px; font-weight: 600; letter-spacing: 1.5px;
    text-transform: uppercase; color: var(--subtext); margin-bottom: 4px;
}
.metric-card .value {
    font-family: 'Syne', sans-serif; font-size: 26px;
    font-weight: 800; color: var(--text);
}
.metric-card .delta { font-size: 11px; color: var(--green); margin-top: 2px; }
.upload-zone {
    border: 2px dashed var(--border); border-radius: 12px;
    padding: 32px 20px; text-align: center; background: #0D1320;
    margin-bottom: 16px;
}
.upload-icon { font-size: 36px; margin-bottom: 8px; }
.upload-text { font-size: 13px; color: var(--subtext); }
.stButton > button {
    background: linear-gradient(135deg, var(--accent1), #3A86FF) !important;
    color: #0A0E17 !important; font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important; font-size: 13px !important;
    letter-spacing: 1px !important; border: none !important;
    border-radius: 8px !important; padding: 12px 0 !important;
    width: 100% !important; text-transform: uppercase !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: var(--panel); border-radius: 8px; padding: 4px;
    gap: 4px; border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background: transparent; color: var(--subtext);
    font-family: 'JetBrains Mono', monospace; font-size: 12px;
    font-weight: 600; border-radius: 6px; padding: 8px 16px;
}
.stTabs [aria-selected="true"] { background: var(--accent1) !important; color: var(--bg) !important; }
hr { border-color: var(--border) !important; }

/* ── Readability Analyzer Styles ─────────────────────────────────────── */
.readability-section {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px 28px;
    margin: 24px 0;
}
.readability-title {
    font-family: 'Syne', sans-serif;
    font-size: 16px; font-weight: 800;
    color: var(--accent1); margin-bottom: 4px;
}
.readability-subtitle {
    font-size: 11px; color: var(--subtext);
    margin-bottom: 20px; letter-spacing: 0.5px;
}
.audience-card {
    border-radius: 12px; padding: 18px 20px;
    margin-bottom: 12px; border: 1px solid var(--border);
    position: relative; overflow: hidden;
}
.audience-card::before {
    content: ''; position: absolute;
    left: 0; top: 0; bottom: 0; width: 4px;
    border-radius: 12px 0 0 12px;
}
.audience-card.layman::before   { background: #4CC9F0; }
.audience-card.student::before  { background: #F77F00; }
.audience-card.professional::before { background: #E63946; }
.audience-card .aud-label {
    font-family: 'Syne', sans-serif;
    font-size: 12px; font-weight: 700;
    letter-spacing: 1.5px; text-transform: uppercase;
    margin-bottom: 6px;
}
.audience-card.layman   .aud-label { color: #4CC9F0; }
.audience-card.student  .aud-label { color: #F77F00; }
.audience-card.professional .aud-label { color: #E63946; }
.audience-card .aud-pct {
    font-family: 'Syne', sans-serif;
    font-size: 32px; font-weight: 800; color: var(--text);
    line-height: 1;
}
.audience-card .aud-desc {
    font-size: 11px; color: var(--subtext); margin-top: 4px;
}
.insight-box {
    border-radius: 12px; padding: 16px 20px;
    margin-top: 16px; font-size: 13px; font-weight: 600;
}
.insight-box.easy    { background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.3); color: #10B981; }
.insight-box.medium  { background: rgba(247,127,0,0.12);  border: 1px solid rgba(247,127,0,0.3);  color: #F77F00; }
.insight-box.hard    { background: rgba(230,57,70,0.12);  border: 1px solid rgba(230,57,70,0.3);  color: #E63946; }
.word-stat-row {
    display: flex; gap: 10px; margin-top: 16px;
}
.word-stat {
    flex: 1; background: #0D1320; border: 1px solid var(--border);
    border-radius: 8px; padding: 10px 14px; text-align: center;
}
.word-stat .ws-val {
    font-family: 'Syne', sans-serif; font-size: 20px;
    font-weight: 800; color: var(--text);
}
.word-stat .ws-lbl {
    font-size: 9px; color: var(--subtext);
    text-transform: uppercase; letter-spacing: 1px; margin-top: 2px;
}
</style>
""", unsafe_allow_html=True)

# ── Colour constants ──────────────────────────────────────────────────────────
COLORS     = {"LAYMAN": "#4CC9F0", "STUDENT": "#F77F00", "PROFESSIONAL": "#E63946"}
LABEL_ORDER = ["LAYMAN", "STUDENT", "PROFESSIONAL"]
DARK_BG    = "#0A0E17"
PANEL_BG   = "#111827"
GRID_CLR   = "#1F2937"
TEXT_CLR   = "#E2E8F0"

plt.rcParams.update({
    "figure.facecolor": DARK_BG, "axes.facecolor": PANEL_BG,
    "axes.edgecolor": GRID_CLR,  "axes.labelcolor": TEXT_CLR,
    "axes.titlecolor": TEXT_CLR, "xtick.color": "#64748B",
    "ytick.color": "#64748B",    "text.color": TEXT_CLR,
    "grid.color": GRID_CLR,      "grid.linestyle": "--",
    "grid.alpha": 0.4,           "font.family": "monospace",
})


# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENT READABILITY ANALYZER — helpers
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text_from_upload(uploaded_file):
    """Extract plain text from a .txt or .pdf upload."""
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        try:
            import pdfplumber
            uploaded_file.seek(0)
            with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
                return "\n".join(
                    page.extract_text() or "" for page in pdf.pages
                )
        except ImportError:
            try:
                import PyPDF2
                uploaded_file.seek(0)
                reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
                return "\n".join(
                    page.extract_text() or "" for page in reader.pages
                )
            except ImportError:
                st.error("📦 Install `pdfplumber` or `PyPDF2` to read PDFs: `pip install pdfplumber`")
                return ""
    else:
        uploaded_file.seek(0)
        return uploaded_file.read().decode("utf-8", errors="ignore")


def tokenize_words(text):
    """Simple word tokenizer — lowercase alpha tokens only."""
    return [w.lower() for w in re.findall(r"[a-zA-Z]+", text) if len(w) >= 2]


def analyse_document_with_model(raw_text: str, filename: str):
    """
    Run the uploaded document text through the REAL trained KNN model pipeline.
    Uses the same feature_engineering + model_training modules as the main GMM section.
    Steps:
      1. Tokenize text → word frequency counter
      2. filter_words (same filters as pipeline)
      3. extract_all_features (Zipf, syllables, BERT similarity, WordNet depth ...)
      4. classify_all_words via trained KNN + zipf fallback
      5. Return per-category counts and percentages
    """
    import tempfile, config, json
    from feature_engineering import (
        read_corpus, filter_words, extract_all_features, get_feature_matrix
    )
    from model_training import (
        train_classifier, classify_all_words, auto_validate_labels,
        create_vocabulary_datasets
    )

    # Load seed words
    with open(config.SEED_FILE, encoding="utf-8") as f:
        seeds = json.load(f)

    # Write text to a temp file so read_corpus() can process it
    tmpdir = tempfile.mkdtemp()
    safe_name = re.sub(r"[^\w.]", "_", filename)
    tmp_path = os.path.join(tmpdir, safe_name if safe_name.endswith(".txt") else safe_name + ".txt")
    with open(tmp_path, "w", encoding="utf-8") as fout:
        fout.write(raw_text)

    documents, word_freq = read_corpus(tmpdir)
    total_tokens = sum(word_freq.values())

    if total_tokens < 20:
        raise ValueError("Document too short for model analysis (< 20 tokens).")

    # Filter words using same pipeline filters
    word_freq_filtered = filter_words(
        word_freq  = word_freq,
        documents  = documents,
        seed_words = seeds,
        min_freq   = 1,                       # relax freq for single-doc analysis
        min_len    = config.MIN_WORD_LENGTH,
        sigma      = config.ZIPF_SIGMA_FILTER,
        cap_ratio  = config.CAP_RATIO_THRESH,
        min_cap    = config.MIN_CAP_COUNT,
    )

    if len(word_freq_filtered) < 3:
        raise ValueError(
            f"Only {len(word_freq_filtered)} words survived filtering. "
            "The document may be too short or contain mostly stopwords."
        )

    # Extract features (includes BERT similarity to seed words)
    df_feat = extract_all_features(word_freq_filtered, documents, seeds)
    X, feature_cols = get_feature_matrix(df_feat)

    # Train KNN on seed words (same as main pipeline)
    clf, scaler, X_scaled, label_map, thresholds = train_classifier(
        df=df_feat, seed_words=seeds, feature_cols=feature_cols, k=config.KNN_K
    )

    # Classify all words using KNN + auto zipf fallback
    labels, probs = classify_all_words(
        df=df_feat, clf=clf, scaler=scaler, X_scaled=X_scaled,
        feature_cols=feature_cols, thresholds=thresholds,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
    )

    # Auto-validate labels using seed statistics
    labels = auto_validate_labels(df_feat, labels, thresholds, sigma=2.0)

    # Count per category
    total         = len(labels)
    layman_count  = int((labels == "LAYMAN").sum())
    student_count = int((labels == "STUDENT").sum())
    prof_count    = int((labels == "PROFESSIONAL").sum())

    layman_pct  = 100 * layman_count  / max(total, 1)
    student_pct = 100 * student_count / max(total, 1)
    prof_pct    = 100 * prof_count    / max(total, 1)

    avg_conf = float(probs.max(axis=1).mean())

    # Sample top words per category for display
    sample_words = {}
    for lbl in ["LAYMAN", "STUDENT", "PROFESSIONAL"]:
        mask = labels == lbl
        if mask.any():
            conf_scores = probs[mask].max(axis=1)
            word_list   = df_feat["word"].values[mask]
            top_idx     = conf_scores.argsort()[::-1][:8]
            sample_words[lbl] = list(word_list[top_idx])
        else:
            sample_words[lbl] = []

    return {
        "total":              total,
        "layman_count":       layman_count,
        "student_count":      student_count,
        "professional_count": prof_count,
        "layman_pct":         layman_pct,
        "student_pct":        student_pct,
        "professional_pct":   prof_pct,
        "avg_confidence":     avg_conf,
        "sample_words":       sample_words,
        "model_used":         True,
    }


def compute_flesch_score(text):
    """Compute Flesch Reading Ease score."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    words = tokenize_words(text)
    if not sentences or not words:
        return None

    def count_syllables(word):
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        prev_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        if word.endswith("e") and count > 1:
            count -= 1
        return max(1, count)

    total_sentences = len(sentences)
    total_words     = len(words)
    total_syllables = sum(count_syllables(w) for w in words)

    if total_sentences == 0 or total_words == 0:
        return None

    asl = total_words / total_sentences          # avg sentence length
    asw = total_syllables / total_words          # avg syllables per word
    score = 206.835 - 1.015 * asl - 84.6 * asw
    return round(max(0, min(100, score)), 1)


def flesch_grade_label(score):
    if score is None:
        return "N/A", "—"
    if score >= 90:
        return "Very Easy", "5th grade"
    elif score >= 80:
        return "Easy", "6th grade"
    elif score >= 70:
        return "Fairly Easy", "7th grade"
    elif score >= 60:
        return "Standard", "8–9th grade"
    elif score >= 50:
        return "Fairly Difficult", "High school"
    elif score >= 30:
        return "Difficult", "College"
    else:
        return "Very Difficult", "Professional"


def readability_insight(layman_pct, student_pct, professional_pct):
    """Return CSS class + message for the insight box."""
    if layman_pct >= 55:
        return "easy", "✅ This content is easily understood by a general audience (layman-friendly)."
    elif layman_pct + student_pct >= 70:
        return "medium", "📘 This content is well-suited for students and educated readers."
    elif professional_pct >= 40:
        return "hard", "⚠️ This content is complex — best suited for professionals or domain experts."
    elif student_pct >= 50:
        return "medium", "📘 This content is moderately complex — suitable for college-level readers."
    else:
        return "medium", "📊 Mixed complexity — content spans multiple audience levels."


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE  — uses new KNN classifier pipeline
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_pipeline_modules():
    import config
    from feature_engineering import (
        read_corpus, filter_words, extract_all_features, get_feature_matrix
    )
    from model_training import (
        train_classifier, classify_all_words, auto_validate_labels,
        analyze_clusters, create_vocabulary_datasets
    )
    return (config, read_corpus, filter_words, extract_all_features,
            get_feature_matrix, train_classifier, classify_all_words,
            auto_validate_labels, analyze_clusters, create_vocabulary_datasets)


def run_pipeline(uploaded_files, n_components=3):
    (config, read_corpus, filter_words, extract_all_features,
     get_feature_matrix, train_classifier, classify_all_words,
     auto_validate_labels, analyze_clusters,
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

    clf, scaler, X_scaled, label_map, thresholds = train_classifier(
        df=df, seed_words=seeds, feature_cols=feature_cols, k=config.KNN_K
    )

    labels, probs = classify_all_words(
        df=df, clf=clf, scaler=scaler, X_scaled=X_scaled,
        feature_cols=feature_cols, thresholds=thresholds,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
    )

    labels = auto_validate_labels(df, labels, thresholds, sigma=2.0)
    comp_stats = analyze_clusters(df, labels)

    unique_labels = sorted(set(labels))
    int_label_map = {i: l for i, l in enumerate(unique_labels)}
    str2int = {l: i for i, l in enumerate(unique_labels)}
    int_labels = np.array([str2int[l] for l in labels])

    datasets = create_vocabulary_datasets(df, labels, probs)

    for lbl in list(datasets.keys()):
        d = datasets[lbl]
        if not isinstance(d, pd.DataFrame):
            if isinstance(d, list):
                datasets[lbl] = pd.DataFrame(d) if d and isinstance(d[0], dict) \
                                 else pd.DataFrame({"word": d, "confidence": 1.0})
            else:
                datasets[lbl] = pd.DataFrame(columns=["word", "confidence"])

    from sklearn.metrics import silhouette_score
    try:
        sil = silhouette_score(X_scaled, labels)
    except Exception:
        sil = 0.0

    class _FakeGMM:
        def aic(self, X): return 0.0
        def bic(self, X): return 0.0

    return X_scaled, df, int_labels, probs, comp_stats, int_label_map, datasets, _FakeGMM(), sil


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
    r   = pca.fit_transform(normalized)
    colors = [COLORS.get(label_map.get(c, ""), "#888") for c in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(r[:, 0], r[:, 1], c=colors, alpha=0.55, s=16, linewidths=0)

    from matplotlib.patches import Ellipse
    for cid, lbl in label_map.items():
        mask = labels == cid
        if mask.sum() < 5:
            continue
        pts  = r[mask]
        mean = pts.mean(axis=0)
        cov  = np.cov(pts.T)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        w, h  = 2 * 2.0 * np.sqrt(np.abs(vals))
        ell   = Ellipse(xy=mean, width=w, height=h, angle=angle,
                        color=COLORS.get(lbl, "#888"), alpha=0.12, zorder=0)
        ell_e = Ellipse(xy=mean, width=w, height=h, angle=angle,
                        fill=False, edgecolor=COLORS.get(lbl, "#888"),
                        linewidth=1.5, alpha=0.6, zorder=1)
        ax.add_patch(ell)
        ax.add_patch(ell_e)
        ax.plot(*mean, "+", color=COLORS.get(lbl, "#888"),
                markersize=10, markeredgewidth=2)

    ax.set_title(f"Word Difficulty Clusters  (K={len(label_map)})", fontsize=13, pad=10)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend(handles=_legend(label_map), fontsize=9, title="Cluster")
    ax.grid(True)
    fig.tight_layout()
    return fig


def chart_distribution(datasets):
    counts = {l: len(datasets[l]) for l in LABEL_ORDER if l in datasets}
    lbls   = list(counts.keys())
    vals   = list(counts.values())
    clrs   = [COLORS[l] for l in lbls]

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
        try:
            from scipy.stats import gaussian_kde
            data = max_probs[mask]
            xs   = np.linspace(0, 1, 200)
            kde  = gaussian_kde(data, bw_method=0.15)
            ys   = kde(xs)
            ax.fill_between(xs, ys, alpha=0.25, color=COLORS[lbl])
            ax.plot(xs, ys, color=COLORS[lbl], linewidth=2, label=lbl)
        except Exception:
            pass

    ax.set_title("Density Plots", fontsize=12)
    ax.set_xlabel("Max Posterior Probability")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.grid(True)
    fig.tight_layout()
    return fig


def chart_correlation(df):
    num_df = df.select_dtypes(include=[np.number])
    corr   = num_df.corr()
    cmap   = LinearSegmentedColormap.from_list("rg", ["#E63946", PANEL_BG, "#4CC9F0"])
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
                               "avg_word_length", "avg_wordnet_depth"]
                 if any(k in v for v in comp_stats.values())]
    if len(stat_keys) < 3:
        return None
    N      = len(stat_keys)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(figsize=(5, 5),
                           subplot_kw={"polar": True, "facecolor": PANEL_BG})
    fig.patch.set_facecolor(DARK_BG)
    for lbl, stats in comp_stats.items():
        color = COLORS.get(lbl, "#888")
        raw   = [stats.get(k, 0) for k in stat_keys]
        all_v = [[s.get(k, 0) for s in comp_stats.values()] for k in stat_keys]
        normed = [(v - min(av)) / (max(av) - min(av) + 1e-9)
                  for v, av in zip(raw, all_v)]
        vals = normed + normed[:1]
        ax.plot(angles, vals, color=color, linewidth=2, label=lbl)
        ax.fill(angles, vals, color=color, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([k.replace("avg_", "").replace("_", " ").title()
                        for k in stat_keys], fontsize=9, color=TEXT_CLR)
    ax.set_yticklabels([])
    ax.set_title("Component Radar", fontsize=12, pad=16, color=TEXT_CLR)
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.grid(color=GRID_CLR, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def chart_readability_donut(layman_pct, student_pct, professional_pct):
    """Donut chart showing audience breakdown."""
    sizes  = [layman_pct, student_pct, professional_pct]
    labels = ["LAYMAN", "STUDENT", "PROFESSIONAL"]
    clrs   = [COLORS[l] for l in labels]
    # Filter out zero slices for cleaner display
    filtered = [(s, l, c) for s, l, c in zip(sizes, labels, clrs) if s > 0]
    if not filtered:
        return None
    fsizes, flabels, fclrs = zip(*filtered)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    wedges, texts, autotexts = ax.pie(
        fsizes, labels=flabels, colors=fclrs,
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"width": 0.55, "edgecolor": DARK_BG, "linewidth": 2},
        textprops={"color": TEXT_CLR, "fontsize": 9},
    )
    for at in autotexts:
        at.set_color(DARK_BG)
        at.set_fontsize(9)
        at.set_fontweight("bold")
    ax.set_title("Audience Breakdown", fontsize=12, pad=12)
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
# LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

sidebar, main = st.columns([1, 2.8], gap="large")

with sidebar:
    st.markdown('<div class="section-title">1. Data & Model Setup</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="upload-zone">
      <div class="upload-icon">📄</div>
      <div class="upload-text">Drag & Drop .TXT files or browse</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload .txt files", type=["txt"],
        accept_multiple_files=True, label_visibility="collapsed",
    )

    if uploaded:
        st.markdown(f"**{len(uploaded)} file(s) loaded:**")
        for f in uploaded:
            size_kb = len(f.getvalue()) / 1024
            st.markdown(f"&nbsp;&nbsp;📄 `{f.name}` — {size_kb:.1f} KB")

    st.markdown("---")
    st.markdown('<div class="section-title">GMM Configuration</div>',
                unsafe_allow_html=True)

    k_clusters  = st.slider("Number of Clusters (K)", 2, 5, 3)
    cov_type    = st.selectbox("Covariance Type", ["full", "tied", "diag", "spherical"])
    init_method = st.selectbox("Initialization", ["kmeans", "k-means++", "random"])
    max_iter    = st.number_input("Max Iterations", 50, 500, 100, step=50)

    st.markdown("---")
    run_btn = st.button("▶  RUN GMM ANALYSIS")

with main:
    st.markdown('<div class="section-title">2. Visualization & Analysis Results</div>',
                unsafe_allow_html=True)

    if "results" not in st.session_state:
        st.session_state.results = None

    if run_btn:
        if not uploaded:
            st.warning("⚠️  Please upload at least one .txt file first.")
        else:
            with st.spinner("Running pipeline..."):
                progress = st.progress(0, text="Loading modules...")
                try:
                    progress.progress(20, "Extracting features...")
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

    if st.session_state.results is not None:
        normalized, df, labels, probs, comp_stats, label_map, datasets, gmm, sil = \
            st.session_state.results

        difficulty_labels = [label_map.get(c, "?") for c in labels]
        max_probs = probs.max(axis=1)
        avg_conf  = max_probs.mean()
        counts    = {l: len(datasets.get(l, [])) for l in LABEL_ORDER}

        # ── Metric cards ──────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-card">
            <div class="label">Total Words</div>
            <div class="value">{len(df):,}</div>
            <div class="delta">↑ processed</div>
          </div>
          <div class="metric-card">
            <div class="label">LAYMAN</div>
            <div class="value">{counts.get('LAYMAN', 0):,}</div>
            <div class="delta">{100*counts.get('LAYMAN',0)/max(len(df),1):.1f}%</div>
          </div>
          <div class="metric-card">
            <div class="label">STUDENT</div>
            <div class="value">{counts.get('STUDENT', 0):,}</div>
            <div class="delta">{100*counts.get('STUDENT',0)/max(len(df),1):.1f}%</div>
          </div>
          <div class="metric-card">
            <div class="label">PROFESSIONAL</div>
            <div class="value">{counts.get('PROFESSIONAL', 0):,}</div>
            <div class="delta">{100*counts.get('PROFESSIONAL',0)/max(len(df),1):.1f}%</div>
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

        st.pyplot(chart_pca(normalized, labels, label_map), use_container_width=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="section-title">Cluster Distribution</div>',
                        unsafe_allow_html=True)
            st.pyplot(chart_distribution(datasets), use_container_width=True)
        with c2:
            st.markdown('<div class="section-title">Density Plots</div>',
                        unsafe_allow_html=True)
            try:
                st.pyplot(chart_density(probs, labels, label_map), use_container_width=True)
            except Exception:
                st.info("scipy required for density plots")
        with c3:
            st.markdown('<div class="section-title">Component Radar</div>',
                        unsafe_allow_html=True)
            r = chart_radar(comp_stats, label_map)
            if r:
                st.pyplot(r, use_container_width=True)

        st.markdown("---")
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Feature Analysis", "📋 Word Explorer",
            "📈 Model Metrics",    "💾 Export",
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
                "Select cluster", LABEL_ORDER, horizontal=True,
                label_visibility="collapsed",
            )
            if selected_label in datasets:
                ds = datasets[selected_label].copy()
                word_col = next((c for c in ["word", "Word", "token"]
                                 if c in ds.columns), ds.columns[0])
                conf_col = next((c for c in ["confidence", "conf", "score"]
                                 if c in ds.columns),
                                ds.columns[-1] if len(ds.columns) > 1 else word_col)
                search = st.text_input("🔍 Search words", placeholder="Type to filter...")
                if search:
                    ds = ds[ds[word_col].astype(str).str.contains(
                        search, case=False, na=False)]
                display_cols = [word_col] if word_col == conf_col else [word_col, conf_col]
                st.dataframe(
                    ds[display_cols].rename(columns={word_col: "Word", conf_col: "Confidence"})
                      .sort_values("Confidence", ascending=False)
                      .reset_index(drop=True),
                    use_container_width=True, height=400,
                )
                st.caption(f"Showing {len(ds):,} words")

        with tab3:
            st.markdown("**Model Evaluation Metrics**")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Silhouette Score", f"{sil:.4f}")
                st.metric("Avg Confidence",   f"{avg_conf:.4f}")
            with m2:
                st.metric("Total Words",  f"{len(df):,}")
                st.metric("Clusters (K)", len(label_map))
            with m3:
                st.metric("Features Used", normalized.shape[1])
                st.metric("Seed Words",
                          sum(len(v) for v in
                              __import__('json').load(
                                  open(__import__('config').SEED_FILE,
                                       encoding='utf-8')).values()))

            st.markdown("**Per-Cluster Statistics**")
            rows = []
            for lbl, stat in sorted(comp_stats.items()):
                rows.append({
                    "Cluster":        lbl,
                    "Words":          stat.get("size", 0),
                    "Avg Zipf":       f"{stat.get('avg_zipf_score', 0):.3f}",
                    "Avg Domain":     f"{stat.get('avg_domain_specificity', 0):.3f}",
                    "Avg Length":     f"{stat.get('avg_word_length', 0):.2f}",
                    "Avg Confidence": f"{stat.get('avg_confidence', avg_conf):.3f}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        with tab4:
            st.markdown("**Download vocabulary datasets**")
            for lbl in LABEL_ORDER:
                ds_exp = datasets.get(lbl)
                if ds_exp is not None and len(ds_exp) > 0:
                    csv = ds_exp.to_csv(index=False)
                    st.download_button(
                        label=f"⬇  Download {lbl.lower()}_vocabulary.csv  ({len(ds_exp):,} words)",
                        data=csv,
                        file_name=f"{lbl.lower()}_vocabulary.csv",
                        mime="text/csv",
                        key=f"dl_{lbl}",
                    )

    else:
        st.markdown("""
        <div style="border: 2px dashed #1F2937; border-radius: 16px;
                    padding: 80px 40px; text-align: center; background: #0D1320;">
            <div style="font-size: 64px; margin-bottom: 16px;">🔬</div>
            <div style="font-family: 'Syne', sans-serif; font-size: 20px;
                        font-weight: 800; color: #4CC9F0; margin-bottom: 8px;">
                Ready to Analyse
            </div>
            <div style="font-size: 13px; color: #64748B; max-width: 360px; margin: 0 auto;">
                Upload one or more <code>.txt</code> files in the left panel,
                configure your settings, then click <strong>RUN GMM ANALYSIS</strong>.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 📄 DOCUMENT READABILITY ANALYZER — uses already-uploaded files from sidebar
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div class="readability-section">
  <div class="readability-title">📄 Document Readability Analyzer</div>
  <div class="readability-subtitle">
    USES YOUR TRAINED KNN MODEL (BERT + ZIPF + WORDNET FEATURES) — SAME MODEL AS THE GMM PIPELINE ABOVE
  </div>
</div>
""", unsafe_allow_html=True)

if "ra_results" not in st.session_state:
    st.session_state.ra_results = None

# ── Auto-run when files are uploaded, or show Analyse button ─────────────────
if uploaded:
    # File selector if multiple files uploaded
    if len(uploaded) > 1:
        file_names = [f.name for f in uploaded]
        selected_fname = st.selectbox(
            "Select file to analyse readability:",
            file_names,
            key="ra_file_select",
        )
        doc_file = next(f for f in uploaded if f.name == selected_fname)
    else:
        doc_file = uploaded[0]

    ra_info_col, ra_btn_col = st.columns([3, 1])
    with ra_info_col:
        size_kb = len(doc_file.getvalue()) / 1024
        st.markdown(f"""
        <div style="background:#0D1320; border:1px solid #1F2937; border-radius:8px;
                    padding:10px 16px; font-size:12px; color:#64748B; margin-bottom:8px;">
            📄 <strong style="color:#E2E8F0;">{doc_file.name}</strong>
            &nbsp;·&nbsp; {size_kb:.1f} KB
            &nbsp;·&nbsp; <span style="color:#4CC9F0;">Ready to analyse</span>
        </div>
        """, unsafe_allow_html=True)
    with ra_btn_col:
        analyze_btn = st.button("🔍  ANALYSE READABILITY", key="analyze_readability")

    if analyze_btn:
        with st.spinner("🔬 Running your trained KNN model on the document..."):
            try:
                raw_text = extract_text_from_upload(doc_file)
                if not raw_text.strip():
                    st.error("Could not extract text. Please check the file.")
                else:
                    # ── Use the REAL trained KNN model pipeline ──────────────
                    result = analyse_document_with_model(raw_text, doc_file.name)

                    flesch           = compute_flesch_score(raw_text)
                    f_label, f_grade = flesch_grade_label(flesch)
                    ins_cls, ins_msg = readability_insight(
                        result["layman_pct"],
                        result["student_pct"],
                        result["professional_pct"],
                    )

                    st.session_state.ra_results = {
                        **result,
                        "flesch":   flesch,
                        "f_label":  f_label,
                        "f_grade":  f_grade,
                        "ins_cls":  ins_cls,
                        "ins_msg":  ins_msg,
                        "raw_text": raw_text,
                        "filename": doc_file.name,
                    }
            except Exception as e:
                st.error(f"Error during analysis: {e}")
                with st.expander("Show traceback"):
                    st.exception(e)

else:
    # No files uploaded yet
    st.markdown("""
    <div style="border: 2px dashed #1F2937; border-radius: 12px;
                padding: 40px 32px; text-align: center; background: #0D1320; margin-top:8px;">
        <div style="font-size: 40px; margin-bottom: 10px;">📑</div>
        <div style="font-family: 'Syne', sans-serif; font-size: 15px;
                    font-weight: 800; color: #4CC9F0; margin-bottom: 6px;">
            No Files Uploaded Yet
        </div>
        <div style="font-size: 12px; color: #64748B; max-width: 340px; margin: 0 auto;">
            Upload <strong>.txt</strong> files using the panel on the left —
            then click <strong>Analyse Readability</strong> to see the audience breakdown.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Results display ───────────────────────────────────────────────────────────
if st.session_state.ra_results is not None:
    R = st.session_state.ra_results

    st.markdown(f"""
    <div style="background:#0D1320; border:1px solid #1F2937; border-radius:8px;
                padding:10px 16px; margin:12px 0; font-size:12px; color:#64748B;">
        ✅ <strong style="color:#10B981;">Analysis complete</strong>
        &nbsp;·&nbsp; 📄 <strong style="color:#E2E8F0;">{R['filename']}</strong>
        &nbsp;·&nbsp; {R['total']:,} words analysed
        &nbsp;·&nbsp; <span style="background:#4CC9F020; color:#4CC9F0; padding:2px 8px;
                    border-radius:4px; border:1px solid #4CC9F040; font-size:10px; font-weight:700;">
            🤖 KNN MODEL
        </span>
        &nbsp;·&nbsp; <span style="color:#10B981;">avg confidence {R.get('avg_confidence', 0):.1%}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Audience cards + donut ────────────────────────────────────────────────
    card_col, donut_col = st.columns([1.4, 1])

    with card_col:
        st.markdown(f"""
        <div class="audience-card layman">
            <div class="aud-label">👤 Layman</div>
            <div class="aud-pct">{R['layman_pct']:.1f}%</div>
            <div class="aud-desc">{R['layman_count']:,} words · general vocabulary</div>
        </div>
        <div class="audience-card student">
            <div class="aud-label">📘 Student</div>
            <div class="aud-pct">{R['student_pct']:.1f}%</div>
            <div class="aud-desc">{R['student_count']:,} words · academic / intermediate</div>
        </div>
        <div class="audience-card professional">
            <div class="aud-label">🏛️ Professional</div>
            <div class="aud-pct">{R['professional_pct']:.1f}%</div>
            <div class="aud-desc">{R['professional_count']:,} words · domain-specific / expert</div>
        </div>
        """, unsafe_allow_html=True)

    with donut_col:
        donut = chart_readability_donut(R["layman_pct"], R["student_pct"], R["professional_pct"])
        if donut:
            st.pyplot(donut, use_container_width=True)

    # ── Sample words per category ────────────────────────────────────────────
    sample = R.get("sample_words", {})
    if any(sample.values()):
        sw1, sw2, sw3 = st.columns(3)
        for col, lbl, clr in zip(
            [sw1, sw2, sw3],
            ["LAYMAN", "STUDENT", "PROFESSIONAL"],
            ["#4CC9F0", "#F77F00", "#E63946"]
        ):
            with col:
                words_str = "  ·  ".join(sample.get(lbl, []))
                col.markdown(f"""
                <div style="background:#0D1320; border:1px solid #1F2937; border-radius:8px;
                            padding:10px 14px; min-height:60px;">
                    <div style="font-size:9px; color:{clr}; font-weight:700;
                                letter-spacing:1.5px; text-transform:uppercase;
                                margin-bottom:6px;">{lbl} — top words</div>
                    <div style="font-size:11px; color:#94A3B8; line-height:1.8;">
                        {words_str if words_str else "—"}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Progress bars ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title" style="margin-top:16px;">Understandability Score</div>',
                unsafe_allow_html=True)

    prog_col1, prog_col2 = st.columns(2)
    with prog_col1:
        st.markdown("**👤 Layman**")
        st.progress(R["layman_pct"] / 100)
        st.caption(f"{R['layman_pct']:.1f}% of words are layman-level")

        st.markdown("**📘 Student**")
        st.progress(R["student_pct"] / 100)
        st.caption(f"{R['student_pct']:.1f}% of words are student-level")

    with prog_col2:
        st.markdown("**🏛️ Professional**")
        st.progress(R["professional_pct"] / 100)
        st.caption(f"{R['professional_pct']:.1f}% of words are professional-level")

        if R["flesch"] is not None:
            st.markdown("**📖 Flesch Reading Ease**")
            st.progress(R["flesch"] / 100)
            st.caption(f"{R['flesch']} — {R['f_label']} ({R['f_grade']})")

    # ── Word stats mini row ───────────────────────────────────────────────────
    st.markdown(f"""
    <div class="word-stat-row">
      <div class="word-stat">
        <div class="ws-val">{R['total']:,}</div>
        <div class="ws-lbl">Total Words</div>
      </div>
      <div class="word-stat">
        <div class="ws-val" style="color:#4CC9F0;">{R['layman_count']:,}</div>
        <div class="ws-lbl">Layman</div>
      </div>
      <div class="word-stat">
        <div class="ws-val" style="color:#F77F00;">{R['student_count']:,}</div>
        <div class="ws-lbl">Student</div>
      </div>
      <div class="word-stat">
        <div class="ws-val" style="color:#E63946;">{R['professional_count']:,}</div>
        <div class="ws-lbl">Professional</div>
      </div>
      <div class="word-stat">
        <div class="ws-val">{R['flesch'] if R['flesch'] is not None else 'N/A'}</div>
        <div class="ws-lbl">Flesch Score</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── AI Insight box ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="insight-box {R['ins_cls']}">
        {R['ins_msg']}
    </div>
    """, unsafe_allow_html=True)

    # ── Text preview ──────────────────────────────────────────────────────────
    with st.expander("📝 View extracted text preview"):
        preview = R["raw_text"][:2000]
        if len(R["raw_text"]) > 2000:
            preview += "\n\n... [truncated — showing first 2000 characters]"
        st.text_area("Extracted Text", preview, height=200, label_visibility="collapsed")