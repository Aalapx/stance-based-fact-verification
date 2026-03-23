import faiss
import json
import spacy
import pickle

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.bm25 import build_bm25


def load_all():

    dense_model = SentenceTransformer("intfloat/e5-large")

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

    index = faiss.read_index("data/fever_hnsw.index")

    with open("data/fever_texts.json", "r") as f:
        sentences = json.load(f)

    tfidf_vectorizer = None
    tfidf_matrix = None

    with open("data/wiki_index.pkl", "rb") as f:
        page_index = pickle.load(f)

    nlp = spacy.load("en_core_web_sm")

    bm25 = build_bm25(sentences)

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
        "nlp": nlp,
        "bm25": bm25
    }