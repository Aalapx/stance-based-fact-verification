# src/model_loader.py

import torch
import faiss
import pickle
import spacy
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification


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