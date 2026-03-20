import numpy as np


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
        if idx < len(sentences):   # 🔥 CRASH FIX
            results.append({"sentence": sentences[idx]})

    return results


# ---------------- HYBRID (TEMP) ----------------

def hybrid_retrieve(
    claim,
    dense_model,
    index,
    sentences,
    tfidf_vectorizer,
    tfidf_matrix,
    top_k=50
):
    return dense_retrieve(
        claim, dense_model, index, sentences, top_k
    )