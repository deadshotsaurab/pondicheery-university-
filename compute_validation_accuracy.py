import pandas as pd

print("\n===== COMPUTING VALIDATION ACCURACY =====\n")

def compute_accuracy(file, group):
    df = pd.read_csv(file)

    total = len(df)

    correct = len(df[df["Grade"] == "High"])
    partial = len(df[df["Grade"] == "Moderate"])

    accuracy = (correct + 0.5 * partial) / total * 100

    print(f"{group} Validation Accuracy: {accuracy:.2f}%")
    return accuracy

layman_acc = compute_accuracy("../validation/validation_layman.csv", "Layman")
student_acc = compute_accuracy("../validation/validation_student.csv", "Student")
professional_acc = compute_accuracy("../validation/validation_professional.csv", "Professional")

overall = (layman_acc + student_acc + professional_acc) / 3

print("\n=======================================")
print(f"OVERALL MODEL VALIDATION ACCURACY: {overall:.2f}%")
print("=======================================\n")