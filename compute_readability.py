import os
import re
import pandas as pd

print("\nSTEP 1: Loading Domain Vocabulary...")

# Load vocabulary from OUTPUT folder
layman = pd.read_csv("output/layman_vocabulary.csv")
student = pd.read_csv("output/student_vocabulary.csv")
professional = pd.read_csv("output/professional_vocabulary.csv")

# Build difficulty dictionary
difficulty_dict = {}

for w in layman["word"]:
    difficulty_dict[w] = 1

for w in student["word"]:
    difficulty_dict[w] = 2

for w in professional["word"]:
    difficulty_dict[w] = 3

print("Total vocabulary loaded:", len(difficulty_dict))


# =========================================================
# TOKENIZER
# =========================================================

def tokenize(text):
    return re.findall(r"\b[a-z]{2,}\b", text.lower())


# =========================================================
# READ DOCUMENTS
# =========================================================

print("\nSTEP 2: Reading Legal Documents...")

files = [os.path.join("data", f) for f in os.listdir("data") if f.endswith(".txt")]

print("Total files:", len(files))


# =========================================================
# COMPUTE READABILITY
# =========================================================

print("\nSTEP 3: Computing Readability Scores...")

results = []

for file in files:

    with open(file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    tokens = tokenize(text)

    total_words = len(tokens)
    if total_words == 0:
        continue

    layman_count = 0
    student_count = 0
    professional_count = 0
    score_sum = 0

    for word in tokens:
        if word in difficulty_dict:
            level = difficulty_dict[word]
            score_sum += level

            if level == 1:
                layman_count += 1
            elif level == 2:
                student_count += 1
            else:
                professional_count += 1

    avg_score = score_sum / total_words

    # Classify readability
    if avg_score <= 1.5:
        readability = "EASY"
    elif avg_score <= 2.2:
        readability = "MODERATE"
    else:
        readability = "COMPLEX"

    results.append({
        "file_name": file,
        "total_words": total_words,
        "avg_difficulty_score": round(avg_score, 3),
        "layman_%": round(layman_count / total_words * 100, 2),
        "student_%": round(student_count / total_words * 100, 2),
        "professional_%": round(professional_count / total_words * 100, 2),
        "readability_level": readability
    })


# =========================================================
# SAVE RESULTS
# =========================================================

df = pd.DataFrame(results)

output_file = "output/document_readability_scores.csv"
df.to_csv(output_file, index=False)

print("\nSTEP 4: Readability Analysis Complete!")
print(df)

print("\nResults saved at:", output_file)