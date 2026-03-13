import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def clean_evidence(text):
    return text.split("\t")[0]


# --------------------------------------------------
# ENTITY PAGE RETRIEVAL
# --------------------------------------------------

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


# --------------------------------------------------
# DENSE RETRIEVAL
# --------------------------------------------------

def dense_retrieve(claim, dense_model, index, sentences, top_k=50):

    claim_embedding = dense_model.encode([claim], convert_to_numpy=True)
    scores, indices = index.search(claim_embedding, top_k)

    results = []
    for idx in indices[0]:
        results.append({"sentence": sentences[idx]})

    return results


# --------------------------------------------------
# TFIDF RETRIEVAL
# --------------------------------------------------

def tfidf_retrieve(claim, tfidf_vectorizer, tfidf_matrix, sentences, top_k=50):

    claim_vec = tfidf_vectorizer.transform([claim])
    similarities = cosine_similarity(claim_vec, tfidf_matrix)[0]
    top_indices = similarities.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({"sentence": sentences[idx]})

    return results


# --------------------------------------------------
# HYBRID RETRIEVAL
# --------------------------------------------------

def hybrid_retrieve(
    claim,
    dense_model,
    index,
    sentences,
    tfidf_vectorizer,
    tfidf_matrix,
    top_k=50
):

    dense_results = dense_retrieve(
        claim, dense_model, index, sentences, top_k
    )

    tfidf_results = tfidf_retrieve(
        claim, tfidf_vectorizer, tfidf_matrix, sentences, top_k
    )

    combined = {r["sentence"]: r for r in dense_results + tfidf_results}

    return list(combined.values())