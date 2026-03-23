import numpy as np
from rank_bm25 import BM25Okapi


def clean_evidence(text):
    return text.replace("passage: ", "").split("\t")[0]


# ---------------- ENTITY ----------------

def entity_page_retrieve(claim, nlp, page_index, max_sentences=25):

    doc = nlp(claim)
    entities = [ent.text.replace(" ", "_") for ent in doc.ents]

    candidates = []

    for ent in entities:
        if ent in page_index:
            sentence_dict = page_index[ent]
            count = 0
            for _, sentence in sentence_dict.items():
                if sentence.strip() == "":
                    continue
                candidates.append({"sentence": sentence})
                count += 1
                if count >= max_sentences:
                    break

    return candidates


# ---------------- DENSE ----------------

def dense_retrieve(claim, dense_model, index, sentences, top_k=50):

    claim_embedding = dense_model.encode(
        ["query: " + claim],
        convert_to_numpy=True
    )

    scores, indices = index.search(claim_embedding, top_k)

    results = []
    for idx in indices[0]:
        if idx < len(sentences):
            results.append({"sentence": sentences[idx]})

    return results


# ---------------- BM25 ----------------

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

    results = []
    for idx in top_indices:
        results.append({"sentence": sentences[idx]})

    return results