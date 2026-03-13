import streamlit as st
import torch
import numpy as np
import torch.nn.functional as F
import pandas as pd

from src.model_loader import load_all
from src.pipeline import verify_claim
resources = load_all()

dense_model = resources["dense_model"]
reranker_tokenizer = resources["reranker_tokenizer"]
reranker_model = resources["reranker_model"]
stance_tokenizer = resources["stance_tokenizer"]
stance_model = resources["stance_model"]
index = resources["index"]
sentences = resources["sentences"]
tfidf_vectorizer = resources["tfidf_vectorizer"]
tfidf_matrix = resources["tfidf_matrix"]
page_index = resources["page_index"]
nlp = resources["nlp"]




torch.set_grad_enabled(False)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(page_title="Stance Verification System", layout="wide")
st.title("🧠 Stance-based Fact Verification System")
st.markdown("Dense + TFIDF Retrieval • Cross-Encoder Reranking • Multi-Evidence Stance")


# --------------------------------------------------
# STANCE
# --------------------------------------------------


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
            result = verify_claim(
                claim_input,
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
                stance_model
            )

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