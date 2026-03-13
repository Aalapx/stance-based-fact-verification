import streamlit as st
import torch
import faiss
import pickle
import numpy as np
import spacy
import torch.nn.functional as F
import pandas as pd

from src.retrieval import (
    clean_evidence,
    entity_page_retrieve,
    dense_retrieve,
    tfidf_retrieve,
    hybrid_retrieve
)

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity

torch.set_grad_enabled(False)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(page_title="Stance Verification System", layout="wide")
st.title("🧠 Stance-based Fact Verification System")
st.markdown("Dense + TFIDF Retrieval • Cross-Encoder Reranking • Multi-Evidence Stance")

# --------------------------------------------------
# LOAD MODELS (CACHED)
# --------------------------------------------------

@st.cache_resource
def load_all():

    dense_model = SentenceTransformer("all-MiniLM-L6-v2")

    reranker_tokenizer = AutoTokenizer.from_pretrained(
        "models/fever_reranker_model_final"
    )
    reranker_model = AutoModelForSequenceClassification.from_pretrained(
        "models/fever_reranker_model_final"
    )
    reranker_model.eval()

    stance_tokenizer = AutoTokenizer.from_pretrained(
        "models/fever_finetuned_model"
    )
    stance_model = AutoModelForSequenceClassification.from_pretrained(
        "models/fever_finetuned_model"
    )
    stance_model.eval()

    index = faiss.read_index("data/faiss_index_1M_fixed.bin")

    with open("data/sentences_1M.pkl", "rb") as f:
        sentences = pickle.load(f)

    with open("data/tfidf_index_1M.pkl", "rb") as f:
        tfidf_data = pickle.load(f)

    tfidf_vectorizer = tfidf_data["vectorizer"]
    tfidf_matrix = tfidf_data["matrix"]

    with open("data/wiki_index.pkl", "rb") as f:
        page_index = pickle.load(f)

    nlp = spacy.load("en_core_web_sm")

    return (
        dense_model,
        reranker_tokenizer,
        reranker_model,
        stance_tokenizer,
        stance_model,
        index,
        sentences,
        tfidf_vectorizer,
        tfidf_matrix,
        page_index,
        nlp
    )


(
    dense_model,
    reranker_tokenizer,
    reranker_model,
    stance_tokenizer,
    stance_model,
    index,
    sentences,
    tfidf_vectorizer,
    tfidf_matrix,
    page_index,
    nlp
) = load_all()


# --------------------------------------------------
# RERANKING
# --------------------------------------------------

def rerank(claim, candidates, top_k=3):

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

# --------------------------------------------------
# STANCE
# --------------------------------------------------

def joint_evidence_stance(claim, top_ranked):

    # Use only best evidence for stability
    best = top_ranked[0]
    sentence = clean_evidence(best[0]["sentence"])

    inputs = stance_tokenizer(
        claim,
        sentence,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )

    with torch.no_grad():
        outputs = stance_model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
    pred_id = int(np.argmax(probs))
    confidence = float(np.max(probs))
    label = stance_model.config.id2label[pred_id]

    return label, confidence, [sentence]

###
def detect_relation_type(claim):

    doc = nlp(claim)

    for token in doc:
        if token.lemma_ in ["bear", "born"]:
            return "birth"
        if token.lemma_ == "capital":
            return "capital"
        if token.lemma_ in ["make", "made"]:
            return "composition"

    return "generic"

###
def relation_filter(claim, candidates):

    relation_type = detect_relation_type(claim)
    filtered = []

    claim_doc = nlp(claim)
    claim_entities = [ent.text.lower() for ent in claim_doc.ents]

    for c in candidates:

        text = clean_evidence(c["sentence"])
        text_lower = text.lower()

        # Require entity overlap
        # Require ALL named entities from claim to appear
        if not all(ent in text_lower for ent in claim_entities):
            continue

        # Birth claims
        if relation_type == "birth":
            if "born" not in text_lower:
                continue

        # Capital claims
        elif relation_type == "capital":
            if "capital" not in text_lower:
                continue

        # Composition claims
        elif relation_type == "composition":
            if not any(word in text_lower for word in ["made", "composed", "consists"]):
                continue

        filtered.append(c)

    return filtered

# --------------------------------------------------
# VERIFY
# --------------------------------------------------

def verify_claim(claim):

    # STEP 1: Retrieval
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

    # STEP 2: Rerank (Top 1)
    top_ranked = rerank(claim, candidates, top_k=1)

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

    # -------- Semantic similarity guard --------
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

    # STEP 3: Stance classification
    inputs = stance_tokenizer(
        claim,
        best_sentence,
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

    pred_id = int(np.argmax(probs))
    stance = label_map[pred_id]
    confidence = float(np.max(probs))

    prob_dict = {
        "SUPPORTS": float(probs[0]),
        "REFUTES": float(probs[1]),
        "NOT ENOUGH INFO": float(probs[2])
    }

    # Confidence gating
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
# --------------------------------------------------
# UI
# --------------------------------------------------
# --------------------------------------------------
# DEMO EXAMPLES
# --------------------------------------------------

support_examples = [
    "Paris is the capital of France.",
    "Barack Obama was born in Hawaii."
]

refute_examples = [
    "Paris is the capital of Germany.",
    "Barack Obama was born in Kenya."
]

nei_examples = [
    "Tilda Swinton is a professional chess player.",
    "Elon Musk is a chess grandmaster."
]

if "demo_claim" not in st.session_state:
    st.session_state.demo_claim = ""

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Load SUPPORT Example"):
        st.session_state.demo_claim = np.random.choice(support_examples)

with col2:
    if st.button("Load REFUTE Example"):
        st.session_state.demo_claim = np.random.choice(refute_examples)

with col3:
    if st.button("Load NEI Example"):
        st.session_state.demo_claim = np.random.choice(nei_examples)

claim_input = st.text_input(
    "Enter a claim to verify",
    value=st.session_state.demo_claim
)

if st.button("Verify"):

    if claim_input.strip() == "":
        st.warning("Please enter a claim.")
    else:
        with st.spinner("Processing..."):
            result = verify_claim(claim_input)

        stance = result["stance"]
        confidence = result["confidence"]
        evidences = result["evidences"]
        probabilities = result["probabilities"]

        # ---------- VERDICT ----------
        if stance == "SUPPORTS":
            st.success(f"SUPPORTS (Confidence: {round(confidence,3)})")
        elif stance == "REFUTES":
            st.error(f"REFUTES (Confidence: {round(confidence,3)})")
        else:
            st.warning(f"NOT ENOUGH INFO (Confidence: {round(confidence,3)})")

        # ---------- EVIDENCE ----------
        if evidences:
            st.subheader("Top Evidence")
            for i, ev in enumerate(evidences, 1):
                st.info(f"{i}. {ev}")

        # ---------- PROBABILITY GRAPH ----------
        st.subheader("📊 Stance Probability Distribution")

        prob_df = pd.DataFrame({
            "Stance": list(probabilities.keys()),
            "Probability": list(probabilities.values())
        })

        st.bar_chart(prob_df.set_index("Stance"))