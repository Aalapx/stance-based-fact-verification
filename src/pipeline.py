from src.reranker import rerank
from src.retrieval import dense_retrieve, bm25_retrieve, clean_evidence
from sklearn.metrics.pairwise import cosine_similarity


def verify_claim(
    claim,
    nlp,
    page_index,
    dense_model,
    index,
    sentences,
    reranker_tokenizer,
    reranker_model,
    stance_tokenizer,
    stance_model,
    bm25
):

    # -----------------------------
    # STEP 1: RETRIEVAL
    # -----------------------------
    dense_candidates = dense_retrieve(
        claim, dense_model, index, sentences, top_k=50
    )

    bm25_candidates = bm25_retrieve(
        claim, bm25, sentences, top_k=50
    )

    candidates = (dense_candidates + bm25_candidates)[:100]

    if len(candidates) == 0:
        return {
            "stance": "NOT ENOUGH INFO",
            "confidence": 1.0,
            "evidences": [],
            "probabilities": {}
        }

    # -----------------------------
    # STEP 2: RERANK
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
            "probabilities": {}
        }

    # -----------------------------
    # STEP 3: CONTRADICTION FILTER (NEW)
    # -----------------------------
    best_sentence = clean_evidence(top_ranked[0][0]["sentence"])

    # Quick lexical check (cheap but powerful)
    claim_words = set(claim.lower().split())
    evidence_words = set(best_sentence.lower().split())

    overlap = len(claim_words & evidence_words) / (len(claim_words) + 1e-5)

    if overlap < 0.2:
        return {
            "stance": "NOT ENOUGH INFO",
            "confidence": 0.0,
            "evidences": [best_sentence],
            "probabilities": {}
        }

    # -----------------------------
    # STEP 3: SIMILARITY GUARD (BACK)
    # -----------------------------
    claim_emb = dense_model.encode(
        ["query: " + claim],
        convert_to_numpy=True
    )

    evidence_emb = dense_model.encode(
        ["passage: " + best_sentence],
        convert_to_numpy=True
    )

    similarity = cosine_similarity(claim_emb, evidence_emb)[0][0]

    if similarity < 0.55:   # IMPORTANT VALUE
        return {
            "stance": "NOT ENOUGH INFO",
            "confidence": float(similarity),
            "evidences": [best_sentence],
            "probabilities": {}
        }

    # -----------------------------
    # STEP 4: STANCE
    # -----------------------------
    from src.stance import classify_stance

    stance, confidence, prob_dict = classify_stance(
        claim,
        best_sentence,
        stance_tokenizer,
        stance_model
    )

    # -----------------------------
    # STEP 5: CONFIDENCE GATING (BACK)
    # -----------------------------
    if confidence < 0.85:
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