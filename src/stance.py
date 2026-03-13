import torch


def classify_stance(claim, evidence, stance_tokenizer, stance_model):

    inputs = stance_tokenizer(
        claim,
        evidence,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )

    with torch.no_grad():
        outputs = stance_model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1).numpy()[0]

    label_map = {
        0: "SUPPORTS",
        1: "REFUTES",
        2: "NOT ENOUGH INFO"
    }

    pred_id = int(probs.argmax())
    stance = label_map[pred_id]
    confidence = float(probs[pred_id])

    prob_dict = {
        "SUPPORTS": float(probs[0]),
        "REFUTES": float(probs[1]),
        "NOT ENOUGH INFO": float(probs[2])
    }

    return stance, confidence, prob_dict