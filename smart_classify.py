"""
smart_classify.py
Final vocabulary classifier using multi-signal approach:
  1. Hard stopword removal (went, far, yes, viz, etc.)
  2. Proper noun detection (names, places)
  3. Domain specificity filter (som, hari = rare but not legal)
  4. KNN seed similarity (primary semantic signal)
  5. Zipf as tiebreaker (scientifically justified)
"""

import os
import json
import re
import pandas as pd
import numpy as np
from wordfreq import zipf_frequency

# ─── CONFIG ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = "output"
SEED_FILE  = "seed_words.json"

# Words to ALWAYS remove — stopwords, trivial words, punctuation artifacts
STOPWORDS = set("""
a an the is are was were be been being have has had do does did will would
could should may might shall can need dare ought used to of in on at by for
from with about above below between into through during before after under
over then when where why how all each every few more most other some such
no nor not only own same so than too very just but and or if as it its
itself they them their there this that these those i me my we our you your
he she his her him we us his its
went far side try yes no came says actually later also even still yet
already always never often sometimes usually generally mostly mainly
first second third next last also too even back up down out off over
again further once around here there
""".split())

# Noise patterns — not real vocabulary
NOISE_PATTERNS = [
    r"^\d+$",           # pure numbers
    r"^[a-z]{1,2}$",   # single/double letter
    r"'s$",             # possessives
    r"^viz$",           # Latin abbreviation
    r"^ibid$",
    r"^etc$",
    r"^re$",
    r"^vs$",
]

# Known proper nouns from Indian legal corpus
PROPER_NOUNS = set("""
william singh kumar sharma iyer iyengar maharaja calcutta bombay madras
bengal punjab mysore kerala gujarat rajasthan kolhapur visakhapatnam
ramalakshmi ranganayaki chunderbutti girdhari mahalaxmi golap sushilabala
bhupendra jnanendra dwarka ghose bose sundaram sukhchain travancore
krishna rama hari som vesco krishnaswami ramprasad shankar narayanan
gopal venkat murali laxmi saraswati durga lakshmi parvati ganesh shiva
vishnu brahma indra arjuna yudhisthira bhima nakula sahadeva draupadi
""".split())


def is_noise(word: str) -> bool:
    """Return True if the word should be excluded entirely."""
    w = word.lower().strip()
    if w in STOPWORDS:
        return True
    if w in PROPER_NOUNS:
        return True
    for pat in NOISE_PATTERNS:
        if re.match(pat, w):
            return True
    # Very short and low domain specificity → likely stopword not caught
    if len(w) <= 3 and zipf_frequency(w, 'en') > 4.0:
        return True
    return False


def classify_word(word: str, row: pd.Series) -> str:
    """
    Multi-signal classification.
    Signals used (in priority order):
      1. Hard rules (stopword/noise → discard)
      2. Seed similarity gap (sim_professional - sim_layman)
      3. Domain specificity (≥ 2.0 = legal domain)
      4. Zipf score (fallback boundary)
    """
    zipf   = float(row.get("zipf_score", 3.0))
    dom    = float(row.get("domain_specificity", 0.0))
    sim_l  = float(row.get("sim_layman", 0.4))
    sim_s  = float(row.get("sim_student", 0.4))
    sim_p  = float(row.get("sim_professional", 0.4))
    length = len(word)

    # Compute gaps
    gap_PL = sim_p - sim_l   # positive → more professional
    gap_SL = sim_s - sim_l   # positive → more student

    # ── PROFESSIONAL signals ──────────────────────────────────────────────────
    # Strong professional: very low zipf AND domain-specific
    if zipf < 2.5 and dom >= 1.5:
        return "PROFESSIONAL"
    # Strong professional: high pro similarity AND long word
    if gap_PL > 0.08 and length >= 8:
        return "PROFESSIONAL"
    # Low zipf + long + domain specific
    if zipf < 3.2 and length >= 9 and dom >= 0.5:
        return "PROFESSIONAL"

    # ── LAYMAN signals ────────────────────────────────────────────────────────
    # Very common word: zipf ≥ 4.8 AND not domain-specific
    if zipf >= 4.8 and dom < 1.0:
        return "LAYMAN"
    # Common + layman-similar
    if zipf >= 4.5 and sim_l > sim_p and dom < 1.5:
        return "LAYMAN"

    # ── STUDENT (default for middle-ground) ───────────────────────────────────
    # Default: zipf 3.2–4.8 = student level
    if 3.2 <= zipf < 4.8:
        # But pull back to LAYMAN if clearly not domain-specific
        if zipf >= 4.3 and dom < 0.5 and gap_PL < 0.0:
            return "LAYMAN"
        # Pull to PROFESSIONAL if domain-specific and long
        if dom >= 2.0 and length >= 7 and zipf < 3.8:
            return "PROFESSIONAL"
        return "STUDENT"

    # Final fallback
    if zipf >= 4.8:
        return "LAYMAN"
    elif zipf >= 3.2:
        return "STUDENT"
    else:
        return "PROFESSIONAL"


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load existing classified words (from train_classifier.py output) ──────
    files = {
        "LAYMAN":       os.path.join(OUTPUT_DIR, "layman_vocabulary.csv"),
        "STUDENT":      os.path.join(OUTPUT_DIR, "student_vocabulary.csv"),
        "PROFESSIONAL": os.path.join(OUTPUT_DIR, "professional_vocabulary.csv"),
    }

    dfs = []
    for label, path in files.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["original_label"] = label
            dfs.append(df)

    if not dfs:
        print("ERROR: No vocabulary CSVs found in output/. Run train_classifier.py first.")
        return

    all_words = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(all_words)} total word entries")

    # ── Remove duplicates (keep highest confidence) ───────────────────────────
    if "confidence" in all_words.columns:
        all_words = all_words.sort_values("confidence", ascending=False)
    all_words = all_words.drop_duplicates(subset="word", keep="first")
    print(f"After dedup: {len(all_words)} unique words")

    # ── Remove noise and proper nouns ─────────────────────────────────────────
    mask_keep = ~all_words["word"].apply(is_noise)
    removed = (~mask_keep).sum()
    all_words = all_words[mask_keep].copy()
    print(f"After noise removal: {len(all_words)} words  (removed {removed})")

    # ── Re-classify every word using multi-signal logic ───────────────────────
    new_labels = []
    for _, row in all_words.iterrows():
        new_labels.append(classify_word(row["word"], row))
    all_words["label"] = new_labels

    # ── Split into final CSVs ─────────────────────────────────────────────────
    results = {"LAYMAN": [], "STUDENT": [], "PROFESSIONAL": []}
    for _, row in all_words.iterrows():
        results[row["label"]].append(row)

    total = len(all_words)
    print("\nFINAL DISTRIBUTION:")
    for lbl, rows in results.items():
        df_out = pd.DataFrame(rows).drop(columns=["original_label", "label"], errors="ignore")
        df_out = df_out.sort_values("zipf_score", ascending=False) if "zipf_score" in df_out.columns else df_out
        pct = 100 * len(df_out) / total
        avg_z = df_out["zipf_score"].mean() if "zipf_score" in df_out.columns else 0
        sample = df_out["word"].head(10).tolist()
        print(f"  {lbl:15s}: {len(df_out):4d} words ({pct:5.1f}%)  avg_zipf={avg_z:.2f}")
        print(f"    Sample: {sample}")

        out_path = os.path.join(OUTPUT_DIR, f"{lbl.lower()}_vocabulary.csv")
        df_out.to_csv(out_path, index=False)

    print(f"\nCSVs saved to {OUTPUT_DIR}/")

    # ── Sanity checks ─────────────────────────────────────────────────────────
    print("\nSANITY CHECK:")
    l_df = pd.DataFrame(results["LAYMAN"])
    p_df = pd.DataFrame(results["PROFESSIONAL"])

    definitely_pro = ['testamentary','estoppel','mandamus','certiorari','subrogation',
                      'adjudication','indemnification','executrix','alienations','amicus']
    definitely_lay = ['wife','husband','son','daughter','death','land','pay',
                      'village','money','family','child','birth']

    l_words = set(l_df["word"].tolist()) if len(l_df) > 0 else set()
    p_words = set(p_df["word"].tolist()) if len(p_df) > 0 else set()

    wrongly_in_layman = [w for w in definitely_pro if w in l_words]
    wrongly_in_prof   = [w for w in definitely_lay if w in p_words]

    print(f"  Pro words wrongly in LAYMAN:  {wrongly_in_layman if wrongly_in_layman else 'NONE ✓'}")
    print(f"  Lay words wrongly in PROF:    {wrongly_in_prof   if wrongly_in_prof   else 'NONE ✓'}")


if __name__ == "__main__":
    run()