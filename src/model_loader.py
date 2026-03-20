import torch
import faiss
import json
import spacy
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_all():

    # 🔥 E5 MODEL
    dense_model = SentenceTransformer("intfloat/e5-large")

    # 🔥 RERANKER
    reranker_tokenizer = AutoTokenizer.from_pretrained(
        "models/fever_reranker_model_final"
    )
    reranker_model = AutoModelForSequenceClassification.from_pretrained(
        "models/fever_reranker_model_final"
    )
    reranker_model.eval()

    # 🔥 STANCE
    stance_tokenizer = AutoTokenizer.from_pretrained(
        "models/fever_finetuned_model"
    )
    stance_model = AutoModelForSequenceClassification.from_pretrained(
        "models/fever_finetuned_model"
    )
    stance_model.eval()

    # 🔥 NEW INDEX
    index = faiss.read_index("data/fever_hnsw.index")

    # 🔥 NEW SENTENCES
    with open("data/fever_texts.json", "r") as f:
        sentences = json.load(f)

    # ❌ TEMP DISABLED
    tfidf_vectorizer = None
    tfidf_matrix = None

    # 🔥 PAGE INDEX
    with open("data/wiki_index.pkl", "rb") as f:
        page_index = pickle.load(f)

    # 🔥 NLP
    nlp = spacy.load("en_core_web_sm")

    return {
        "dense_model": dense_model,
        "reranker_tokenizer": reranker_tokenizer,
        "reranker_model": reranker_model,
        "stance_tokenizer": stance_tokenizer,
        "stance_model": stance_model,
        "index": index,
        "sentences": sentences,
        "tfidf_vectorizer": tfidf_vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "page_index": page_index,
        "nlp": nlp
    }