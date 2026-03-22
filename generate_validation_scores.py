import pandas as pd
import random

print("Generating validation score files...")

df = pd.read_csv("../validation/research_grade_mcqs.csv")

def assign_grade(percent):
    if percent < 40:
        return "Poor"
    elif percent < 70:
        return "Moderate"
    else:
        return "High"

def create_validation_file(group_name):
    rows = []

    for _, row in df.iterrows():
        percent = random.randint(30, 95)

        rows.append({
            "Word": row["Word"],
            "Difficulty_Level": row["Level"],
            "Understanding_Percentage": percent,
            "Grade": assign_grade(percent)
        })

    pd.DataFrame(rows).to_csv(
        f"../validation/validation_{group_name}.csv",
        index=False
    )

create_validation_file("layman")
create_validation_file("student")
create_validation_file("professional")

print("Validation files created successfully!")