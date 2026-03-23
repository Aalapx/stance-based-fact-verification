from rank_bm25 import BM25Okapi


def build_bm25(sentences):
    tokenized = [s.lower().split() for s in sentences]
    return BM25Okapi(tokenized)


def bm25_retrieve(claim, bm25, sentences, top_k=50):
    tokenized_query = claim.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    return [{"sentence": sentences[i]} for i in top_indices]