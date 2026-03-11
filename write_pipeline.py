import os

code = """import os, json, numpy as np
import config
from feature_engineering import read_corpus, filter_words, extract_all_features, get_feature_matrix
from model_training import train_classifier, classify_all_words, auto_validate_labels, analyze_clusters, create_vocabulary_datasets, save_vocabulary_datasets, save_model
from visualization import create_all_visualizations

config.validate()

with open(config.SEED_FILE, encoding="utf-8") as f:
    seeds = json.load(f)
print("Seeds:", {k: len(v) for k, v in seeds.items()})

documents, word_freq = read_corpus(config.DATA_DIR)

word_freq = filter_words(
    word_freq, documents, seeds,
    config.MIN_CORPUS_FREQ, config.MIN_WORD_LENGTH,
    config.ZIPF_SIGMA_FILTER, config.CAP_RATIO_THRESH, config.MIN_CAP_COUNT
)

df = extract_all_features(word_freq, documents, seeds)
X, feature_cols = get_feature_matrix(df)
print("Feature matrix:", X.shape)

clf, scaler, X_scaled, label_map, thresholds = train_classifier(df, seeds, feature_cols, config.KNN_K)

labels, probs = classify_all_words(df, clf, scaler, X_scaled, feature_cols, thresholds, config.CONFIDENCE_THRESHOLD)

labels = auto_validate_labels(df, labels, thresholds, sigma=2.0)

comp_stats = analyze_clusters(df, labels)
total = len(labels)
print("FINAL DISTRIBUTION:")
for lbl, stat in sorted(comp_stats.items()):
    pct = 100 * stat["size"] / total
    z = stat.get("avg_zipf_score", 0)
    print(f"  {lbl}: {stat['size']} words ({pct:.1f}%) avg_zipf={z:.2f}")

datasets = create_vocabulary_datasets(df, labels, probs)
save_vocabulary_datasets(datasets, config.OUTPUT_DIR)
save_model(clf, scaler, thresholds, feature_cols, config.OUTPUT_DIR)

create_all_visualizations(X_scaled, df, labels, probs, datasets, comp_stats, {v: v for v in set(labels)}, config.OUTPUT_DIR)

print("Done!")
"""

with open("run_pipeline.py", "w", encoding="utf-8") as f:
    f.write(code)

print("run_pipeline.py created successfully")