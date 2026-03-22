import os, json, numpy as np
import config
from feature_engineering import filter_words, extract_all_features, get_feature_matrix
from model_training import train_classifier, classify_all_words, auto_validate_labels, analyze_clusters, create_vocabulary_datasets, save_vocabulary_datasets, save_model
from visualization import create_all_visualizations

# ── Validate config ────────────────────────────────────────────────────────
config.validate()

# ── Load seeds ─────────────────────────────────────────────────────────────
with open(config.SEED_FILE, encoding="utf-8") as f:
    seeds = json.load(f)
print("Seeds:", {k: len(v) for k, v in seeds.items()})

# ── Read corpus from ALL 3 subfolders ──────────────────────────────────────
import re
import glob
from collections import Counter

JUDGEMENT_DIR    = os.path.join(config.DATA_DIR, "IN-Ext_judgement")
GROUND_TRUTH_DIR = os.path.join(config.DATA_DIR, "A1_Ground truth")
LOCKNKEY_DIR     = os.path.join(config.DATA_DIR, "A2_LocknKey")

all_folders = [JUDGEMENT_DIR, GROUND_TRUTH_DIR, LOCKNKEY_DIR]

documents  = []
all_tokens = []

for folder in all_folders:
    paths = glob.glob(os.path.join(folder, "*.txt"))
    for p in paths:
        try:
            text = open(p, encoding="utf-8", errors="ignore").read()
        except Exception:
            text = open(p, encoding="latin-1", errors="ignore").read()
        documents.append(text)
        tokens = re.findall(r"[a-zA-Z]+", text)
        all_tokens.extend(tokens)

word_freq = Counter(all_tokens)
print(f"STEP 1: Read {len(documents)} document(s) from 3 folders.")
print(f"  - IN-Ext_judgement : {len(glob.glob(os.path.join(JUDGEMENT_DIR, '*.txt')))} files")
print(f"  - A1_Ground truth  : {len(glob.glob(os.path.join(GROUND_TRUTH_DIR, '*.txt')))} files")
print(f"  - A2_LocknKey      : {len(glob.glob(os.path.join(LOCKNKEY_DIR, '*.txt')))} files")

# ── Filter words ───────────────────────────────────────────────────────────
word_freq = filter_words(
    word_freq, documents, seeds,
    config.MIN_CORPUS_FREQ, config.MIN_WORD_LENGTH,
    config.ZIPF_SIGMA_FILTER, config.CAP_RATIO_THRESH, config.MIN_CAP_COUNT
)

# ── Extract features ───────────────────────────────────────────────────────
df = extract_all_features(word_freq, documents, seeds)
X, feature_cols = get_feature_matrix(df)
print("Feature matrix:", X.shape)

# ── Train classifier ───────────────────────────────────────────────────────
clf, scaler, X_scaled, label_map, thresholds = train_classifier(df, seeds, feature_cols, config.KNN_K)

# ── Classify all words ─────────────────────────────────────────────────────
labels, probs = classify_all_words(df, clf, scaler, X_scaled, feature_cols, thresholds, config.CONFIDENCE_THRESHOLD)
labels = auto_validate_labels(df, labels, thresholds, sigma=2.0)

# ── Analyze clusters ───────────────────────────────────────────────────────
comp_stats = analyze_clusters(df, labels)
total = len(labels)
print("FINAL DISTRIBUTION:")
for lbl, stat in sorted(comp_stats.items()):
    pct = 100 * stat["size"] / total
    z   = stat.get("avg_zipf_score", 0)
    print(f"  {lbl}: {stat['size']} words ({pct:.1f}%) avg_zipf={z:.2f}")

# ── Save vocabulary datasets & model ──────────────────────────────────────
datasets = create_vocabulary_datasets(df, labels, probs)
save_vocabulary_datasets(datasets, config.OUTPUT_DIR)
save_model(clf, scaler, thresholds, feature_cols, config.OUTPUT_DIR)

# ── Visualizations ─────────────────────────────────────────────────────────
create_all_visualizations(
    X_scaled, df, labels, probs,
    datasets, comp_stats,
    {v: v for v in set(labels)},
    config.OUTPUT_DIR
)

print("\nDone! All outputs saved to:", config.OUTPUT_DIR)