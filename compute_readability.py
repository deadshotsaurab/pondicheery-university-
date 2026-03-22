import os
import re
import pandas as pd

print("\nSTEP 1: Loading Domain Vocabulary...")
layman       = pd.read_csv("output/layman_vocabulary.csv")
student      = pd.read_csv("output/student_vocabulary.csv")
professional = pd.read_csv("output/professional_vocabulary.csv")

difficulty_dict = {}
for w in layman["word"]:       difficulty_dict[w] = 1
for w in student["word"]:      difficulty_dict[w] = 2
for w in professional["word"]: difficulty_dict[w] = 3
print("Total vocabulary loaded:", len(difficulty_dict))

# =========================================================
# TOKENIZER
# =========================================================
def tokenize(text):
    return re.findall(r"\b[a-z]{2,}\b", text.lower())

# =========================================================
# READ DOCUMENTS FROM ALL 3 SUBFOLDERS
# =========================================================
print("\nSTEP 2: Reading Legal Documents...")

FOLDERS = {
    "IN-Ext_judgement" : os.path.join("data", "IN-Ext_judgement"),
    "A1_Ground truth"  : os.path.join("data", "A1_Ground truth"),
    "A2_LocknKey"      : os.path.join("data", "A2_LocknKey"),
}

files = []
for folder_name, folder_path in FOLDERS.items():
    for f in os.listdir(folder_path):
        if f.endswith(".txt"):
            files.append({
                "path"        : os.path.join(folder_path, f),
                "case_id"     : os.path.splitext(f)[0],
                "source_type" : folder_name
            })

print(f"Total files: {len(files)}")
for folder_name, folder_path in FOLDERS.items():
    count = sum(1 for fi in files if fi["source_type"] == folder_name)
    print(f"  - {folder_name}: {count} files")

# =========================================================
# COMPUTE READABILITY
# =========================================================
print("\nSTEP 3: Computing Readability Scores...")

results = []
for file_info in files:
    with open(file_info["path"], "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    tokens      = tokenize(text)
    total_words = len(tokens)
    if total_words == 0:
        continue

    layman_count       = 0
    student_count      = 0
    professional_count = 0
    score_sum          = 0

    for word in tokens:
        if word in difficulty_dict:
            level = difficulty_dict[word]
            score_sum += level
            if level == 1:   layman_count += 1
            elif level == 2: student_count += 1
            else:            professional_count += 1

    avg_score = score_sum / total_words

    if avg_score <= 1.5:   readability = "EASY"
    elif avg_score <= 2.2: readability = "MODERATE"
    else:                  readability = "COMPLEX"

    results.append({
        "case_id"             : file_info["case_id"],
        "source_type"         : file_info["source_type"],
        "file_name"           : file_info["path"],
        "total_words"         : total_words,
        "avg_difficulty_score": round(avg_score, 3),
        "layman_%"            : round(layman_count / total_words * 100, 2),
        "student_%"           : round(student_count / total_words * 100, 2),
        "professional_%"      : round(professional_count / total_words * 100, 2),
        "readability_level"   : readability
    })

# =========================================================
# SAVE RESULTS
# =========================================================
df = pd.DataFrame(results)
output_file = "output/document_readability_scores.csv"
os.makedirs("output", exist_ok=True)
df.to_csv(output_file, index=False)

print("\nSTEP 4: Readability Analysis Complete!")
print(f"\nTotal documents processed: {len(df)}")
print(f"\nReadability distribution:")
print(df.groupby(["source_type", "readability_level"]).size().to_string())
print(f"\nAverage difficulty by source:")
print(df.groupby("source_type")["avg_difficulty_score"].mean().round(3).to_string())
print(f"\nResults saved at: {output_file}")