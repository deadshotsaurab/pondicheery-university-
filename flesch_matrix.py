"""
flesch_matrix.py
================
Reads all 50 Supreme Court verdict .txt files across 3 document types,
computes Flesch Reading Ease score for each, and saves results to CSV.

Run:
    python flesch_matrix.py

Output:
    output/flesch_matrix.csv
"""

import os
import re
import csv

# ── PATHS (auto-detects your folder structure) ─────────────────────
def find_folder(candidates):
    """Return the first existing folder from a list of candidates."""
    for p in candidates:
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(f"Could not find any of: {candidates}")

JUDG_DIR = find_folder([
    os.path.join("data", "IN-Ext_judgement"),
    "IN-Ext_judgement",
])
GT_DIR = find_folder([
    os.path.join("data", "summary", "A1_Ground truth"),
    os.path.join("data", "A1_Ground truth"),
    os.path.join("summary", "A1_Ground truth"),
    "A1_Ground truth",
])
LK_DIR = find_folder([
    os.path.join("data", "summary", "A2_LocknKey"),
    os.path.join("data", "A2_LocknKey"),
    os.path.join("summary", "A2_LocknKey"),
    "A2_LocknKey",
])
OUT_CSV = os.path.join("output", "flesch_matrix.csv")
os.makedirs("output", exist_ok=True)

print(f"  JUDG folder : {JUDG_DIR}")
print(f"  GT folder   : {GT_DIR}")
print(f"  LK folder   : {LK_DIR}\n")

# ── SYLLABLE COUNT ─────────────────────────────────────────────────
VOWELS = set("aeiou")

def count_syllables(word):
    word = word.lower()
    count, prev = 0, False
    for ch in word:
        v = ch in VOWELS
        if v and not prev:
            count += 1
        prev = v
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)

# ── FLESCH READING EASE ────────────────────────────────────────────
def flesch_reading_ease(text):
    """
    Formula: 206.835 - 1.015*(words/sentences) - 84.6*(syllables/words)
    Higher = easier to read.
    Interpretation:
        >= 70  → Easy
        50-69  → Standard
        30-49  → Difficult
        < 30   → Very Difficult
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s.strip()]
    words     = re.findall(r"[a-zA-Z]+", text)

    n_sent = max(1, len(sentences))
    n_word = max(1, len(words))
    n_syl  = sum(count_syllables(w) for w in words)

    score = 206.835 - 1.015 * (n_word / n_sent) - 84.6 * (n_syl / n_word)
    return round(score, 4)

def interpret(score):
    if score >= 70:  return "Easy"
    if score >= 50:  return "Standard"
    if score >= 30:  return "Difficult"
    return "Very Difficult"

# ── COMPUTE ────────────────────────────────────────────────────────
case_files = sorted(os.listdir(JUDG_DIR))
rows = []

print(f"Computing Flesch Reading Ease for {len(case_files)} cases...\n")
print(f"{'Case':<6} {'Case ID':<15} {'Orig Judgement':>15} {'Ground Truth':>13} {'LockNKey':>10}")
print("-" * 65)

for i, fname in enumerate(case_files):
    case_id = fname.replace(".txt", "")

    def read(folder): 
        return open(os.path.join(folder, fname), encoding="utf-8", errors="ignore").read()

    oj = flesch_reading_ease(read(JUDG_DIR))
    gt = flesch_reading_ease(read(GT_DIR))
    lk = flesch_reading_ease(read(LK_DIR))

    rows.append({
        "case_num":              i + 1,
        "case_id":               case_id,
        "flesch_orig_judgement": oj,
        "flesch_ground_truth":   gt,
        "flesch_locknkey":       lk,
        "level_orig_judgement":  interpret(oj),
        "level_ground_truth":    interpret(gt),
        "level_locknkey":        interpret(lk),
    })

    print(f"  {i+1:<4} {case_id:<15} {oj:>15.4f} {gt:>13.4f} {lk:>10.4f}")

# ── SAVE CSV ───────────────────────────────────────────────────────
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"\n✅ Saved: {OUT_CSV}  ({len(rows)} rows)")