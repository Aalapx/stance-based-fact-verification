import json
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Path to TF‑IDF index
TFIDF_INDEX_PATH = "data/tfidf_index.pkl"


def load_tfidf_index():
    """Load TF‑IDF vectorizer, matrix, and sentence list from disk."""
    with open(TFIDF_INDEX_PATH, "rb") as f:
        data = pickle.load(f)

    vectorizer = data["vectorizer"]
    matrix = data["matrix"]
    sentences = data["sentences"]

    return vectorizer, matrix, sentences


def retrieve_evidence(claim, top_k=5):
    """Return top‑k most similar Wikipedia sentences for a claim."""
    vectorizer, matrix, sentences = load_tfidf_index()

    # Convert claim to TF‑IDF vector
    claim_vec = vectorizer.transform([claim])

    # Compute cosine similarity
    similarities = cosine_similarity(claim_vec, matrix)[0]

    # Get indices of top‑k highest similarity scores
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "sentence": sentences[idx],
            "score": float(similarities[idx])
        })

    return results


if __name__ == "__main__":
    test_claim = "Barack Obama was born in Kenya."

    print("\nClaim:", test_claim)
    print("Top retrieved evidence:\n")

    evidence_list = retrieve_evidence(test_claim, top_k=5)

    for i, item in enumerate(evidence_list, 1):
        print(f"{i}. ({item['score']:.4f}) {item['sentence']}")