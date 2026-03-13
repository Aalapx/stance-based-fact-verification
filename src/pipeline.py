from sklearn.metrics.pairwise import cosine_similarity


def verify_claim(
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
    classify_stance,
):
    # -----------------------------
    # STEP 1: Retrieval
    # -----------------------------
    entity_candidates = entity_page_retrieve(claim, nlp, page_index)

    hybrid_candidates = hybrid_retrieve(
        claim,
        dense_model,
        index,
        sentences,
        tfidf_vectorizer,
        tfidf_matrix,
        top_k=50
    )

    candidates = (entity_candidates + hybrid_candidates)[:80]

    if len(candidates) == 0:
        return {
            "stance": "NOT ENOUGH INFO",
            "confidence": 1.0,
            "evidences": [],
            "probabilities": {
                "SUPPORTS": 0.0,
                "REFUTES": 0.0,
                "NOT ENOUGH INFO": 1.0
            }
        }

    # -----------------------------
    # STEP 2: Rerank
    # -----------------------------
    top_ranked = rerank(
        claim,
        candidates,
        reranker_tokenizer,
        reranker_model,
        top_k=1
    )

    if len(top_ranked) == 0:
        return {
            "stance": "NOT ENOUGH INFO",
            "confidence": 1.0,
            "evidences": [],
            "probabilities": {
                "SUPPORTS": 0.0,
                "REFUTES": 0.0,
                "NOT ENOUGH INFO": 1.0
            }
        }

    best_sentence = clean_evidence(top_ranked[0][0]["sentence"])

    # -----------------------------
    # STEP 3: Semantic Similarity Guard
    # -----------------------------
    claim_emb = dense_model.encode([claim], convert_to_numpy=True)
    evidence_emb = dense_model.encode([best_sentence], convert_to_numpy=True)

    similarity = cosine_similarity(claim_emb, evidence_emb)[0][0]

    if similarity < 0.5:
        return {
            "stance": "NOT ENOUGH INFO",
            "confidence": float(similarity),
            "evidences": [best_sentence],
            "probabilities": {
                "SUPPORTS": 0.0,
                "REFUTES": 0.0,
                "NOT ENOUGH INFO": 1.0
            }
        }

    # -----------------------------
    # STEP 4: Stance Classification
    # -----------------------------
    stance, confidence, prob_dict = classify_stance(
        claim,
        best_sentence,
        stance_tokenizer,
        stance_model
    )

    # -----------------------------
    # STEP 5: Confidence Gating
    # -----------------------------
    if stance != "NOT ENOUGH INFO" and confidence < 0.85:
        return {
            "stance": "NOT ENOUGH INFO",
            "confidence": confidence,
            "evidences": [best_sentence],
            "probabilities": prob_dict
        }

    return {
        "stance": stance,
        "confidence": confidence,
        "evidences": [best_sentence],
        "probabilities": prob_dict
    }