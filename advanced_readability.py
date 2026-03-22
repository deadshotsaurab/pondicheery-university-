import os
import re
import pandas as pd

print("\n===== ADVANCED READABILITY ANALYSIS STARTED =====")

# =====================================================
# LOAD VOCABULARY
# =====================================================

layman = pd.read_csv("layman_vocabulary.csv")
student = pd.read_csv("student_vocabulary.csv")
professional = pd.read_csv("professional_vocabulary.csv")

difficulty_dict = {}

for w in layman["word"]:
    difficulty_dict[w] = 1

for w in student["word"]:
    difficulty_dict[w] = 2

for w in professional["word"]:
    difficulty_dict[w] = 3

professional_set = set(professional["word"])

print("Vocabulary loaded:", len(difficulty_dict))


# =====================================================
# TOKENIZERS
# =====================================================

def tokenize_words(text):
    return re.findall(r"\b[a-z]{2,}\b", text.lower())

def tokenize_sentences(text):
    return re.split(r"[.!?]+", text)


# =====================================================
# LEGAL KEYWORDS FOR CITATION DENSITY
# =====================================================

legal_keywords = ["section", "article", "act", "clause", "rule"]


# =====================================================
# PROCESS DOCUMENTS
# =====================================================

files = [f for f in os.listdir(".") if f.endswith(".txt")]

results = []

for file in files:

    with open(file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    words = tokenize_words(text)
    sentences = tokenize_sentences(text)

    total_words = len(words)
    total_sentences = len([s for s in sentences if s.strip() != ""])

    if total_words == 0 or total_sentences == 0:
        continue

    # =================================================
    # FEATURE 1: WORD DIFFICULTY SCORE
    # =================================================

    difficulty_sum = 0
    professional_count = 0

    for w in words:
        if w in difficulty_dict:
            difficulty_sum += difficulty_dict[w]

        if w in professional_set:
            professional_count += 1

    avg_word_difficulty = difficulty_sum / total_words


    # =================================================
    # FEATURE 2: AVG SENTENCE LENGTH
    # =================================================

    avg_sentence_length = total_words / total_sentences


    # =================================================
    # FEATURE 3: CLAUSE DENSITY
    # =================================================

    clause_count = text.count(",") + text.count(";")
    clause_density = clause_count / total_sentences


    # =================================================
    # FEATURE 4: PROFESSIONAL TERM DENSITY
    # =================================================

    professional_density = professional_count / total_words


    # =================================================
    # FEATURE 5: LEGAL CITATION DENSITY
    # =================================================

    citation_count = sum(1 for w in words if w in legal_keywords)
    citation_density = citation_count / total_words


    # =================================================
    # FINAL READABILITY SCORE (HYBRID MODEL)
    # =================================================

    final_score = (
        avg_word_difficulty * 0.4 +
        avg_sentence_length * 0.2 +
        clause_density * 0.1 +
        professional_density * 0.2 +
        citation_density * 0.1
    )


    # CLASSIFY READABILITY

    if final_score < 2:
        level = "EASY"
    elif final_score < 4:
        level = "MODERATE"
    else:
        level = "COMPLEX"


    results.append({
        "file_name": file,
        "avg_word_difficulty": round(avg_word_difficulty, 3),
        "avg_sentence_length": round(avg_sentence_length, 2),
        "clause_density": round(clause_density, 3),
        "professional_density": round(professional_density, 3),
        "citation_density": round(citation_density, 3),
        "final_readability_score": round(final_score, 3),
        "readability_level": level
    })


# =====================================================
# SAVE OUTPUT
# =====================================================

df = pd.DataFrame(results)

df.to_csv("output/advanced_readability_scores.csv", index=False)

print("\n===== ADVANCED ANALYSIS COMPLETE =====")
print(df)

print("\nSaved to: output/advanced_readability_scores.csv")