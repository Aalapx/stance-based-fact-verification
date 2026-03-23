import torch


def rerank(claim, candidates, tokenizer, model, top_k=5):

    sentences = [c["sentence"] for c in candidates]

    inputs = tokenizer(
        [claim] * len(sentences),
        sentences,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.softmax(outputs.logits, dim=1)[:, 1]

    scored = list(zip(candidates, scores.tolist()))

    ranked = sorted(scored, key=lambda x: x[1], reverse=True)

    return ranked[:top_k]