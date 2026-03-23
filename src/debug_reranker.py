from src.model_loader import load_all
from src.retrieval import dense_retrieve, bm25_retrieve, clean_evidence

def debug():

    resources = load_all()

    dense_model = resources["dense_model"]
    index = resources["index"]
    sentences = resources["sentences"]
    bm25 = resources["bm25"]

    reranker_tokenizer = resources["reranker_tokenizer"]
    reranker_model = resources["reranker_model"]

    claims = [
        "Paris is the capital of France",
        "The Moon is made of cheese",
        "Barack Obama was born in Kenya",
        "The Eiffel Tower is in Berlin",
        "Python is a type of snake"
    ]

    for claim in claims:
        print("\n" + "="*60)
        print("CLAIM:", claim)

        dense = dense_retrieve(claim, dense_model, index, sentences, top_k=10)
        bm25_res = bm25_retrieve(claim, bm25, sentences, top_k=10)

        candidates = dense + bm25_res

        texts = [clean_evidence(c["sentence"]) for c in candidates]

        # Rerank
        inputs = reranker_tokenizer(
            [claim]*len(texts),
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            outputs = reranker_model(**inputs)
            scores = F.softmax(outputs.logits, dim=1)[:, 1]

        ranked = sorted(zip(texts, scores.tolist()), key=lambda x: x[1], reverse=True)

        print("\nTOP 5 RERANKED:")
        for text, score in ranked[:5]:
            print(f"{score:.4f} → {text}")


if __name__ == "__main__":
    debug()