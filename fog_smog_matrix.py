"""
fog_smog_matrix.py
==================
Reads all 50 Supreme Court verdict .txt files across 3 document types,
computes Gunning Fog Index and SMOG Index for each, saves results to CSV.

Run:
    python fog_smog_matrix.py

Output:
    output/fog_smog_matrix.csv
"""

import os
import re
import math
import csv

# ── PATHS (auto-detects your folder structure) ─────────────────────
def find_folder(candidates):
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
OUT_CSV = os.path.join("output", "fog_smog_matrix.csv")
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

def is_complex(word):
    """Complex word = 3 or more syllables."""
    return count_syllables(word) >= 3

# ── GUNNING FOG INDEX ──────────────────────────────────────────────
def gunning_fog(text):
    """
    Formula: 0.4 * ((words/sentences) + 100*(complex_words/words))
    Lower = easier. Measures grade level needed to understand text.
    Interpretation:
        <= 12 → Acceptable
        13-17 → Hard
        > 17  → Very Hard
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s.strip()]
    words     = re.findall(r"[a-zA-Z]+", text)

    n_sent    = max(1, len(sentences))
    n_word    = max(1, len(words))
    n_complex = sum(1 for w in words if is_complex(w))

    score = 0.4 * ((n_word / n_sent) + 100 * (n_complex / n_word))
    return round(score, 4)

def fog_level(score):
    if score <= 12: return "Acceptable"
    if score <= 17: return "Hard"
    return "Very Hard"

# ── SMOG INDEX ─────────────────────────────────────────────────────
def smog_index(text):
    """
    Formula: 3 + sqrt(complex_words * 30 / sentences)
    Designed specifically for legal/technical/medical texts.
    Lower = easier.
    Interpretation:
        <= 12 → Acceptable
        13-16 → Hard
        > 16  → Very Hard
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s.strip()]
    words     = re.findall(r"[a-zA-Z]+", text)

    n_sent    = max(1, len(sentences))
    n_complex = sum(1 for w in words if is_complex(w))

    score = 3 + math.sqrt(n_complex * 30 / n_sent)
    return round(score, 4)

def smog_level(score):
    if score <= 12: return "Acceptable"
    if score <= 16: return "Hard"
    return "Very Hard"

# ── COMPUTE ────────────────────────────────────────────────────────
case_files = sorted(os.listdir(JUDG_DIR))
rows = []

print(f"Computing Gunning Fog + SMOG for {len(case_files)} cases...\n")
print(f"{'Case':<6} {'Case ID':<15} {'Fog OJ':>8} {'Fog GT':>8} {'Fog LK':>8}  {'SMOG OJ':>8} {'SMOG GT':>8} {'SMOG LK':>8}")
print("-" * 80)

for i, fname in enumerate(case_files):
    case_id = fname.replace(".txt", "")

    def read(folder):
        return open(os.path.join(folder, fname), encoding="utf-8", errors="ignore").read()

    fog_oj  = gunning_fog(read(JUDG_DIR))
    fog_gt  = gunning_fog(read(GT_DIR))
    fog_lk  = gunning_fog(read(LK_DIR))
    smog_oj = smog_index(read(JUDG_DIR))
    smog_gt = smog_index(read(GT_DIR))
    smog_lk = smog_index(read(LK_DIR))

    rows.append({
        "case_num":              i + 1,
        "case_id":               case_id,
        "fog_orig_judgement":    fog_oj,
        "fog_ground_truth":      fog_gt,
        "fog_locknkey":          fog_lk,
        "fog_level_oj":          fog_level(fog_oj),
        "fog_level_gt":          fog_level(fog_gt),
        "fog_level_lk":          fog_level(fog_lk),
        "smog_orig_judgement":   smog_oj,
        "smog_ground_truth":     smog_gt,
        "smog_locknkey":         smog_lk,
        "smog_level_oj":         smog_level(smog_oj),
        "smog_level_gt":         smog_level(smog_gt),
        "smog_level_lk":         smog_level(smog_lk),
    })

    print(f"  {i+1:<4} {case_id:<15} {fog_oj:>8.4f} {fog_gt:>8.4f} {fog_lk:>8.4f}  "
          f"{smog_oj:>8.4f} {smog_gt:>8.4f} {smog_lk:>8.4f}")

# ── SAVE CSV ───────────────────────────────────────────────────────
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"\n✅ Saved: {OUT_CSV}  ({len(rows)} rows)")