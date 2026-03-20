import os
import json
import pickle
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

WIKI_PATH = "datasets/wiki-pages"
OUTPUT_INDEX = "data/faiss_full_index.bin"
OUTPUT_METADATA = "data/sentences_full_metadata.pkl"

BATCH_SIZE = 64
PRINT_EVERY = 50000


def main():

    print("Loading embedding model (multi-qa-mpnet-base-dot-v1)...")
    model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")

    dimension = 768
    index = faiss.IndexFlatIP(dimension)

    metadata = []
    total_count = 0

    print("Starting full FEVER wiki indexing...")

    for file in os.listdir(WIKI_PATH):

        if not file.endswith(".jsonl"):
            continue

        file_path = os.path.join(WIKI_PATH, file)

        with open(file_path, "r") as f:

            batch_sentences = []
            batch_meta = []

            for line in f:

                data = json.loads(line)
                page_id = data["id"]
                lines = data["lines"]

                for row in lines.split("\n"):

                    if row.strip() == "":
                        continue

                    parts = row.split("\t")

                    if not parts[0].isdigit():
                        continue

                    sentence_id = int(parts[0])
                    sentence_text = parts[1]

                    batch_sentences.append(sentence_text)
                    batch_meta.append({
                        "page_id": page_id,
                        "sentence_id": sentence_id,
                        "fever_id": f"{page_id}_{sentence_id}",
                        "sentence": sentence_text
                    })

                    if len(batch_sentences) == BATCH_SIZE:

                        embeddings = model.encode(
                            batch_sentences,
                            convert_to_numpy=True,
                            show_progress_bar=False
                        )

                        faiss.normalize_L2(embeddings)
                        index.add(embeddings)

                        metadata.extend(batch_meta)
                        total_count += len(batch_meta)

                        batch_sentences = []
                        batch_meta = []

                        if total_count % PRINT_EVERY == 0:
                            print(f"Indexed {total_count} sentences...")

            # Flush remaining batch
            if batch_sentences:

                embeddings = model.encode(
                    batch_sentences,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )

                faiss.normalize_L2(embeddings)
                index.add(embeddings)

                metadata.extend(batch_meta)
                total_count += len(batch_meta)

    print("Total vectors indexed:", index.ntotal)

    print("Saving FAISS index...")
    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, OUTPUT_INDEX)

    print("Saving metadata...")
    with open(OUTPUT_METADATA, "wb") as f:
        pickle.dump(metadata, f)

    print("Full FEVER index build complete.")


if __name__ == "__main__":
    main()