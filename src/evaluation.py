import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.pipeline import verify_claim
from src.retrieval import (
    clean_evidence,
    entity_page_retrieve,
    hybrid_retrieve
)
from src.reranker import rerank
from src.stance import classify_stance


def load_fever_dev(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def evaluate_stance(
    data,
    nlp,
    page_index,
    dense_model,
    index,
    sentences,
    tfidf_vectorizer,
    tfidf_matrix,
    reranker_tokenizer,
    reranker_model,
    stance_tokenizer,
    stance_model
):
    gold_labels = []
    predicted_labels = []

    for item in tqdm(data[:500]):
        claim = item["claim"]
        gold = item["label"]

        result = verify_claim(
            claim,
            nlp,
            page_index,
            dense_model,
            index,
            sentences,
            tfidf_vectorizer,
            tfidf_matrix,
            reranker_tokenizer,
            reranker_model,
            stance_tokenizer,
            stance_model,
            entity_page_retrieve,
            hybrid_retrieve,
            rerank,
            clean_evidence,
            classify_stance
        )

        predicted = result["stance"]

        gold_labels.append(gold)
        predicted_labels.append(predicted)

    acc = accuracy_score(gold_labels, predicted_labels)
    macro_f1 = f1_score(gold_labels, predicted_labels, average="macro")

    report = classification_report(gold_labels, predicted_labels)

    return acc, macro_f1, report


if __name__ == "__main__":

    from src.model_loader import load_all

    print("Loading models and resources...")
    resources = load_all()

    dense_model = resources["dense_model"]
    reranker_tokenizer = resources["reranker_tokenizer"]
    reranker_model = resources["reranker_model"]
    stance_tokenizer = resources["stance_tokenizer"]
    stance_model = resources["stance_model"]
    index = resources["index"]
    sentences = resources["sentences"]
    tfidf_vectorizer = resources["tfidf_vectorizer"]
    tfidf_matrix = resources["tfidf_matrix"]
    page_index = resources["page_index"]
    nlp = resources["nlp"]

    print("Loading FEVER dev set...")
    dev_data = load_fever_dev("datasets/dev.jsonl")

    print("Running evaluation...")
    acc, macro_f1, report = evaluate_stance(
        dev_data,
        nlp,
        page_index,
        dense_model,
        index,
        sentences,
        tfidf_vectorizer,
        tfidf_matrix,
        reranker_tokenizer,
        reranker_model,
        stance_tokenizer,
        stance_model
    )

    print("\n===== RESULTS =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("\nDetailed Report:")
    print(report)