import json
from tqdm import tqdm
from src.model_loader import load_all
from src.retrieval import entity_page_retrieve, hybrid_retrieve


def load_fever_dev(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def compute_recall_at_k(data, nlp, page_index,
                        dense_model, index, sentences,
                        tfidf_vectorizer, tfidf_matrix,
                        k=50, max_samples=500):

    hits = 0
    total = 0

    for item in tqdm(data[:max_samples]):

        if item["verifiable"] == "NOT VERIFIABLE":
            continue

        claim = item["claim"]
        gold_evidence = item["evidence"]

        # Collect gold sentence IDs
        gold_ids = set()
        for group in gold_evidence:
            for ev in group:
                gold_ids.add(ev[0])

        # Retrieve candidates
        entity_candidates = entity_page_retrieve(claim, nlp, page_index)
        hybrid_candidates = hybrid_retrieve(
            claim,
            dense_model,
            index,
            sentences,
            tfidf_vectorizer,
            tfidf_matrix,
            top_k=k
        )

        candidates = (entity_candidates + hybrid_candidates)[:k]

        retrieved_ids = set()
        for c in candidates:
            if "id" in c:
                retrieved_ids.add(c["id"])

        if len(gold_ids.intersection(retrieved_ids)) > 0:
            hits += 1

        total += 1

    return hits / total if total > 0 else 0


if __name__ == "__main__":

    print("Loading resources...")
    resources = load_all()

    dev_data = load_fever_dev("datasets/dev.jsonl")

    recall = compute_recall_at_k(
        dev_data,
        resources["nlp"],
        resources["page_index"],
        resources["dense_model"],
        resources["index"],
        resources["sentences"],
        resources["tfidf_vectorizer"],
        resources["tfidf_matrix"],
        k=50,
        max_samples=500
    )

    print(f"\nRetrieval Recall@50: {recall:.4f}")