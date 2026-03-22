import re
import os
import pandas as pd
from collections import Counter
from wordfreq import zipf_frequency

# =================================================
# 1. READ ALL TXT FILES FROM ALL 3 SUBFOLDERS
# =================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

JUDGEMENT_DIR   = os.path.join(DATA_DIR, "IN-Ext_judgement")
GROUND_TRUTH_DIR = os.path.join(DATA_DIR, "A1_Ground truth")
LOCKNKEY_DIR    = os.path.join(DATA_DIR, "A2_LocknKey")

# Collect all .txt files from all 3 folders
all_txt_files = []
for folder in [JUDGEMENT_DIR, GROUND_TRUTH_DIR, LOCKNKEY_DIR]:
    for f in os.listdir(folder):
        if f.endswith(".txt"):
            all_txt_files.append(os.path.join(folder, f))

print(f"\nReading {len(all_txt_files)} text files")
print(f"  - IN-Ext_judgement : {len(os.listdir(JUDGEMENT_DIR))} files")
print(f"  - A1_Ground truth  : {len(os.listdir(GROUND_TRUTH_DIR))} files")
print(f"  - A2_LocknKey      : {len(os.listdir(LOCKNKEY_DIR))} files")

# =================================================
# 2. TOKENIZER
# =================================================
def tokenize(text):
    return re.findall(r"\b[a-z]{2,}\b", text.lower())

# =================================================
# 3. LOAD DOCUMENTS
# =================================================
documents = {}
for filepath in all_txt_files:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as file:
        documents[filepath] = file.read()

total_docs = len(documents)

# =================================================
# 4. CORPUS STATISTICS
# =================================================
all_words = []
doc_freq = Counter()

for text in documents.values():
    tokens = tokenize(text)
    all_words.extend(tokens)
    doc_freq.update(set(tokens))

word_counts = Counter(all_words)
total_words = sum(word_counts.values())

# =================================================
# 5. WORD CLASSIFICATION ALGORITHM
# =================================================
def classify_word(word, freq):
    zipf = zipf_frequency(word, "en")
    doc_ratio = doc_freq[word] / total_docs
    corpus_importance = freq / total_words

    # Noise / grammar removal
    if freq == 1 and doc_ratio < 0.1:
        return None
    if len(word) <= 2:
        return None
    if zipf >= 6.5:
        return None
    if zipf >= 6.0 and corpus_importance > 0.01:
        return None

    # Domain specificity
    domain_spec = (doc_ratio * freq) / zipf if zipf > 0 else doc_ratio * freq

    # PROFESSIONAL
    if zipf < 4.0 and doc_ratio >= 0.4 and freq >= 5:
        return "professional"
    if domain_spec > 0.015 and zipf < 4.5 and freq >= 4:
        return "professional"

    # STUDENT
    if 4.0 <= zipf < 5.5 and doc_ratio >= 0.25 and freq >= 3:
        return "student"

    # LAYMAN
    if 5.5 <= zipf < 6.0 and freq >= 3:
        return "layman"

    return None

# =================================================
# 6. BUILD VOCABULARIES
# =================================================
layman, student, professional = [], [], []
layman_words, student_words, professional_words = set(), set(), set()

layman_count = student_count = professional_count = 0

for word, freq in word_counts.items():
    label = classify_word(word, freq)
    if label is None:
        continue

    zipf = zipf_frequency(word, "en")
    doc_ratio = doc_freq[word] / total_docs
    corpus_importance = freq / total_words
    domain_spec = (doc_ratio * freq) / zipf if zipf > 0 else doc_ratio * freq

    row = {
        "word": word,
        "frequency": freq,
        "zipf_score": round(zipf, 2),
        "doc_frequency": doc_freq[word],
        "doc_ratio": round(doc_ratio, 3),
        "corpus_importance": round(corpus_importance * 100, 4),
        "domain_specificity": round(domain_spec, 4)
    }

    if label == "layman":
        layman.append(row)
        layman_words.add(word)
        layman_count += freq
    elif label == "student":
        student.append(row)
        student_words.add(word)
        student_count += freq
    else:
        professional.append(row)
        professional_words.add(word)
        professional_count += freq

# =================================================
# 7. SAVE CSV FILES
# =================================================
output_dir = os.path.join(BASE_DIR)

pd.DataFrame(layman).sort_values("frequency", ascending=False)\
    .to_csv(os.path.join(output_dir, "layman_vocabulary.csv"), index=False)

pd.DataFrame(student).sort_values("frequency", ascending=False)\
    .to_csv(os.path.join(output_dir, "student_vocabulary.csv"), index=False)

pd.DataFrame(professional).sort_values("frequency", ascending=False)\
    .to_csv(os.path.join(output_dir, "professional_vocabulary.csv"), index=False)

print("\nVocabulary CSV files saved successfully!")

# =================================================
# 8. PER-FILE UNDERSTANDING ANALYSIS
# =================================================
print("\n" + "="*70)
print("FILE-WISE UNDERSTANDING ANALYSIS")
print("="*70)

for filepath, text in documents.items():
    tokens = tokenize(text)
    if not tokens:
        continue

    total = len(tokens)
    fname = os.path.basename(filepath)
    folder = os.path.basename(os.path.dirname(filepath))

    l = sum(1 for w in tokens if w in layman_words)
    s = sum(1 for w in tokens if w in student_words)
    p = sum(1 for w in tokens if w in professional_words)

    print(f"\n[{folder}] {fname}")
    print(f"  Layman:       {l/total*100:.2f}%")
    print(f"  Student:      {s/total*100:.2f}%")
    print(f"  Professional: {p/total*100:.2f}%")

# =================================================
# 9. OVERALL CORPUS SUMMARY
# =================================================
layman_pct       = layman_count / total_words * 100
student_pct      = student_count / total_words * 100
professional_pct = professional_count / total_words * 100
complexity_score = (professional_pct*3 + student_pct*2 + layman_pct) / 3

print("\n" + "="*70)
print("OVERALL CORPUS SUMMARY")
print("="*70)
print(f"Total documents : {total_docs}")
print(f"Total words     : {total_words:,}")
print(f"Layman          : {layman_pct:.2f}%")
print(f"Student         : {student_pct:.2f}%")
print(f"Professional    : {professional_pct:.2f}%")
print(f"Complexity Score: {complexity_score:.2f}/100")

print("\nCSV FILES SAVED:")
print("  * layman_vocabulary.csv")
print("  * student_vocabulary.csv")
print("  * professional_vocabulary.csv")
print("="*70)