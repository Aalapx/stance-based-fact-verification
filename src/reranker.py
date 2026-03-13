import torch
import torch.nn.functional as F
from src.retrieval import clean_evidence


def rerank(claim, candidates, reranker_tokenizer, reranker_model, top_k=3):

    if len(candidates) == 0:
        return []

    sentences_only = [clean_evidence(c["sentence"]) for c in candidates]

    inputs = reranker_tokenizer(
        [claim] * len(sentences_only),
        sentences_only,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = reranker_model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)[:, 1]

    scored = list(zip(candidates, probs.tolist()))
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:top_k]