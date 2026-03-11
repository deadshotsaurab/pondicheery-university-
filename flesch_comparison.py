import os
import re
import pandas as pd

print("\n===== FLESCH READABILITY ANALYSIS =====")


# =====================================================
# SYLLABLE COUNT FUNCTION
# =====================================================

def count_syllables(word):
    word = word.lower()
    vowels = "aeiouy"
    count = 0

    if len(word) == 0:
        return 0

    if word[0] in vowels:
        count += 1

    for i in range(1, len(word)):
        if word[i] in vowels and word[i - 1] not in vowels:
            count += 1

    if word.endswith("e"):
        count -= 1

    if count == 0:
        count = 1

    return count


# =====================================================
# TOKENIZERS
# =====================================================

def tokenize_words(text):
    return re.findall(r"\b[a-z]{2,}\b", text.lower())


def tokenize_sentences(text):
    return re.split(r"[.!?]+", text)


# =====================================================
# PROCESS DOCUMENTS
# =====================================================

files = [os.path.join("data", f) for f in os.listdir("data") if f.endswith(".txt")]

results = []

for file in files:

    with open(file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    words = tokenize_words(text)
    sentences = tokenize_sentences(text)

    words = [w for w in words if w.strip() != ""]
    sentences = [s for s in sentences if s.strip() != ""]

    total_words = len(words)
    total_sentences = len(sentences)

    if total_words == 0 or total_sentences == 0:
        continue

    # =================================================
    # CALCULATE FLESCH SCORE
    # =================================================

    total_syllables = sum(count_syllables(w) for w in words)

    asl = total_words / total_sentences
    asw = total_syllables / total_words

    flesch_score = 206.835 - (1.015 * asl) - (84.6 * asw)

    # CLASSIFY FLESCH LEVEL

    if flesch_score >= 60:
        level = "EASY"
    elif flesch_score >= 30:
        level = "MODERATE"
    else:
        level = "COMPLEX"

    results.append({
        "file_name": file,
        "flesch_score": round(flesch_score, 2),
        "flesch_level": level
    })


# =====================================================
# SAVE OUTPUT
# =====================================================

df = pd.DataFrame(results)

os.makedirs("output", exist_ok=True)
df.to_csv("output/flesch_scores.csv", index=False)

print("\nFlesch Analysis Complete")
print(df)

print("\nSaved to: output/flesch_scores.csv")