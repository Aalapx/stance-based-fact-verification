from src.model_loader import load_all
from src.retrieval import dense_retrieve, bm25_retrieve

resources = load_all()

claim = "Paris is the capital of France"

print("CLAIM:", claim)

dense_results = dense_retrieve(
    claim,
    resources["dense_model"],
    resources["index"],
    resources["sentences"],
    top_k=5
)

bm25_results = bm25_retrieve(
    claim,
    resources["bm25"],
    resources["sentences"],
    top_k=5
)

print("\n--- DENSE RESULTS ---")
for r in dense_results:
    print(r["sentence"])

print("\n--- BM25 RESULTS ---")
for r in bm25_results:
    print(r["sentence"])