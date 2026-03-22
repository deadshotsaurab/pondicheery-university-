import pandas as pd

print("Formatting validation CSV...")

# Read original MCQ file
df = pd.read_csv("../validation/research_grade_mcqs.csv")

formatted = []

for i, row in df.iterrows():

    options = (
        f"A) {row['Option_A']}\n"
        f"B) {row['Option_B']}\n"
        f"C) {row['Option_C']}\n"
        f"D) {row['Option_D']}"
    )

    formatted.append({
        "Q_No": i + 1,
        "Word": row["Word"],
        "Difficulty_Level": row["Level"],
        "Question": row["Question"],
        "Options": options,
        "Correct_Answer": row["Correct_Answer"],
        "Explanation": row["Explanation"]
    })

new_df = pd.DataFrame(formatted)

# Save clean CSV
output_file = "../validation/validation_questions_clean.csv"
new_df.to_csv(output_file, index=False)

print("✅ Clean formatted CSV created!")
print("Saved at:", output_file)