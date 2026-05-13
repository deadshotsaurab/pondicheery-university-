# ⚖️ Legal Readability Classification using Gaussian Mixture Model (GMM)

An NLP and Machine Learning based system for automatically classifying Supreme Court legal vocabulary into different readability levels using a Semi-Supervised Gaussian Mixture Model (GMM).

The project helps identify whether legal documents are understandable by:
- 👨‍⚖️ Professionals
- 🎓 Students
- 👨‍👩‍👧 General Public (Layman)

---

# 🚀 Features

- ✅ Semi-Supervised Gaussian Mixture Model (GMM)
- ✅ BERT Seed Word Similarity
- ✅ TF-IDF Domain Specificity
- ✅ Zipf Frequency Analysis
- ✅ WordNet Semantic Depth
- ✅ Syllable Complexity Analysis
- ✅ Legal Vocabulary Classification
- ✅ Streamlit Interactive Dashboard
- ✅ PCA Cluster Visualization
- ✅ Readability Analyzer
- ✅ Real-Time Legal Text Analysis

---

# 🧠 Problem Statement

Legal judgments are often difficult for common people to understand because of highly technical vocabulary. This project automatically analyzes legal documents and categorizes words into:

| Category | Meaning |
|----------|----------|
| LAYMAN | Easy/common vocabulary |
| STUDENT | Moderate academic vocabulary |
| PROFESSIONAL | Highly technical legal vocabulary |

This helps evaluate how accessible a legal judgment is for different audiences.

---

# 🏗️ System Architecture

```text
Legal Text Dataset
        ↓
Feature Engineering
(TF-IDF + Zipf + WordNet + Syllables + BERT)
        ↓
Seed Word Anchoring
        ↓
Gaussian Mixture Model (GMM)
        ↓
Vocabulary Classification
        ↓
Readability Analysis
        ↓
Streamlit Dashboard
