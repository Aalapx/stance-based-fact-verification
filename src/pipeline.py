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

    # -------- RETRIEVAL --------
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
        return {"stance": "NOT ENOUGH INFO", "confidence": 1.0, "evidences": [], "probabilities": {}}

    # -------- RERANK --------
    top_ranked = rerank(
        claim,
        candidates,
        reranker_tokenizer,
        reranker_model,
        top_k=1
    )

    best_sentence = clean_evidence(top_ranked[0][0]["sentence"])

    # -------- SIMILARITY GUARD (FIXED) --------
    claim_emb = dense_model.encode(["query: " + claim], convert_to_numpy=True)
    evidence_emb = dense_model.encode(["passage: " + best_sentence], convert_to_numpy=True)

    similarity = cosine_similarity(claim_emb, evidence_emb)[0][0]

    if similarity < 0.5:
        return {"stance": "NOT ENOUGH INFO", "confidence": similarity, "evidences": [best_sentence], "probabilities": {}}

    # -------- STANCE --------
    stance, confidence, prob_dict = classify_stance(
        claim,
        best_sentence,
        stance_tokenizer,
        stance_model
    )

    if stance != "NOT ENOUGH INFO" and confidence < 0.85:
        return {"stance": "NOT ENOUGH INFO", "confidence": confidence, "evidences": [best_sentence], "probabilities": prob_dict}

    return {
        "stance": stance,
        "confidence": confidence,
        "evidences": [best_sentence],
        "probabilities": prob_dict
    }