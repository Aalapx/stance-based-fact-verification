import json
import pickle
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

# Paths
SENTENCE_FILE = "data/wiki_sentences.jsonl"
OUTPUT_INDEX = "data/tfidf_index.pkl"


def load_sentences():
    """Load clean sentence texts from the JSONL file."""
    sentences = []

    with open(SENTENCE_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading sentences"):
            obj = json.loads(line)
            text = obj.get("text", "").strip()

            if text:
                sentences.append(text)

    return sentences


def build_tfidf(sentences):
    """Create TF-IDF vectorizer and matrix."""
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=200_000
    )

    matrix = vectorizer.fit_transform(sentences)

    return vectorizer, matrix


def save_index(vectorizer, matrix, sentences):
    """Save TF-IDF components to disk."""
    with open(OUTPUT_INDEX, "wb") as f:
        pickle.dump(
            {
                "vectorizer": vectorizer,
                "matrix": matrix,
                "sentences": sentences,
            },
            f,
        )



def main():
    print("\nStep 1: Loading clean Wikipedia sentences...")
    sentences = load_sentences()

    print(f"Total sentences loaded: {len(sentences):,}")

    print("\nStep 2: Building TF-IDF vector space (this may take several minutes)...")
    vectorizer, matrix = build_tfidf(sentences)

    print("\nStep 3: Saving TF-IDF index to disk...")
    save_index(vectorizer, matrix, sentences)

    print("\nTF-IDF index successfully created!")
    print(f"Saved to: {OUTPUT_INDEX}")


if __name__ == "__main__":
    main()