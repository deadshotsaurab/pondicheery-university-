"""
seed_similarity.py
BERT-based seed similarity featurizer.
All similarity scores computed automatically from seed_words.json.
Zero hardcoded values.
"""

import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class SeedSimilarityFeaturizer:
    """
    For every word, computes cosine similarity to each seed group centroid.
    Derived features:
        sim_layman        — similarity to LAYMAN seed centroid
        sim_student       — similarity to STUDENT seed centroid
        sim_professional  — similarity to PROFESSIONAL seed centroid
        seed_gap_LP       — sim_layman - sim_professional
        seed_gap_LS       — sim_layman - sim_student
    """

    def __init__(self, model_name: str, seed_words: dict):
        self.model = SentenceTransformer(model_name)
        self.seed_words = {k.lower(): v for k, v in seed_words.items()}
        self._centroids = {}
        self._compute_centroids()

    def _compute_centroids(self):
        print("Computing seed centroids...")
        for label, words in self.seed_words.items():
            valid = [w for w in words if isinstance(w, str) and w.strip()]
            if not valid:
                continue
            embs = self.model.encode(valid, show_progress_bar=False,
                                     normalize_embeddings=True)
            self._centroids[label] = embs.mean(axis=0)
        print("Seed centroids ready.")

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def transform(self, words: list, batch_size: int = 256) -> pd.DataFrame:
        """
        Encode all words in batches and compute similarity to each centroid.
        Returns DataFrame aligned with `words`.
        """
        all_embs = []
        for i in range(0, len(words), batch_size):
            batch = words[i: i + batch_size]
            embs  = self.model.encode(batch, show_progress_bar=False,
                                      normalize_embeddings=True)
            all_embs.append(embs)
            if (i // batch_size) % 6 == 0:
                print(f"  Seed similarity: {i}/{len(words)}")

        all_embs = np.vstack(all_embs)

        rows = []
        for emb in all_embs:
            row = {}
            for label, centroid in self._centroids.items():
                row[f"sim_{label}"] = self._cosine(emb, centroid)
            rows.append(row)

        df = pd.DataFrame(rows)

        # Compute gap features automatically from whatever labels exist
        if "sim_layman" in df.columns and "sim_professional" in df.columns:
            df["seed_gap_LP"] = df["sim_layman"] - df["sim_professional"]
        if "sim_layman" in df.columns and "sim_student" in df.columns:
            df["seed_gap_LS"] = df["sim_layman"] - df["sim_student"]

        print("Seed similarity features added.")
        return df.reset_index(drop=True)


def load_seed_words(seed_file: str) -> dict:
    with open(seed_file, encoding="utf-8") as f:
        return json.load(f)