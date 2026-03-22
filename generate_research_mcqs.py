import pandas as pd
import random
import nltk
from nltk.corpus import wordnet as wn
import os

print("\n===== GENERATING RESEARCH-GRADE MCQs =====")

# Load vocab
layman = pd.read_csv("output/layman_vocabulary.csv")
student = pd.read_csv("output/student_vocabulary.csv")
professional = pd.read_csv("output/professional_vocabulary.csv")

layman["level"] = "Layman"
student["level"] = "Student"
professional["level"] = "Professional"

all_words = pd.concat([layman, student, professional])

sample_words = all_words.sample(n=30, random_state=42)

mcqs = []

def get_definition(word):
    synsets = wn.synsets(word)
    if synsets:
        return synsets[0].definition()
    return None

for _, row in sample_words.iterrows():
    word = row["word"]
    level = row["level"]

    definition = get_definition(word)

    if not definition:
        continue

    # Distractors = definitions of random words
    distractor_defs = []

    random_words = all_words["word"].sample(20)

    for rw in random_words:
        d = get_definition(rw)
        if d and d != definition:
            distractor_defs.append(d)
        if len(distractor_defs) == 3:
            break

    if len(distractor_defs) < 3:
        continue

    options = distractor_defs + [definition]
    random.shuffle(options)

    correct = ["A","B","C","D"][options.index(definition)]

    mcqs.append({
        "Word": word,
        "Level": level,
        "Question": f"What is the meaning of '{word}'?",
        "Option_A": options[0],
        "Option_B": options[1],
        "Option_C": options[2],
        "Option_D": options[3],
        "Correct_Answer": correct,
        "Explanation": definition
    })

os.makedirs("output", exist_ok=True)

pd.DataFrame(mcqs).to_csv("output/research_grade_mcqs.csv", index=False)

print("\n✅ Research-Grade MCQs Generated Successfully!")