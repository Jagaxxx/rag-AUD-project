# Manages the vector database (FAISS) for similarity search and interfaces with the LLM reader to generate answers based on retrieved contexts.

import re

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import json
from pathlib import Path

# class BM25Retriever:
#     """
#     Baseline retrieval model using the BM25 algorithm.
#     """
#     def __init__(self, corpus: list[dict]):
#         self.corpus = corpus
#         tokenized_corpus = [doc["text"].lower().split() for doc in corpus]
#         self.bm25 = BM25Okapi(tokenized_corpus)

#     def search(self, query: str, top_k: int = 3) -> list[dict]:
#         """
#         Retrieves the top-k most relevant paragraphs based on BM25 scores.
#         """
#         tokenized_query = query.lower().split()
#         return self.bm25.get_top_n(tokenized_query, self.corpus, n=top_k)


# class TFIDFRetriever:
#     """
#     Baseline retrieval model using TF-IDF and Cosine Similarity.
#     """
#     def __init__(self, corpus: list[dict]):
#         self.corpus = corpus
#         self.texts = [doc["text"] for doc in corpus]
    
#         self.vectorizer = TfidfVectorizer(stop_words="english")
#         self.corpus_matrix = self.vectorizer.fit_transform(self.texts)

#     def search(self, query: str, top_k: int = 3) -> list[dict]:
#         """
#         Retrieves the top-k most relevant paragraphs based on cosine similarity of TF-IDF vectors.
#         """
#         query_matrix = self.vectorizer.transform([query])
#         similarities = cosine_similarity(query_matrix, self.corpus_matrix).flatten()
        
#         top_indices = np.argsort(similarities)[::-1][:top_k]
        
#         results = []
#         for idx in top_indices:
#             doc = self.corpus[idx].copy()
#             doc["score"] = float(similarities[idx])
#             results.append(doc)
            
#         return results


def tokenize(text: str) -> list[str]:
    """Lowercase words without English stop words"""
    return [w for w in re.findall(r"\w+", text.lower()) if w not in ENGLISH_STOP_WORDS]


class BM25Retriever:
    """Baseline retrieval model using the BM25 algorithm."""

    def __init__(self, corpus: list[dict]):
        self.corpus = corpus
        self.bm25 = BM25Okapi([tokenize(doc["text"]) for doc in corpus])

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Return the top-k documents with their BM25 'score'."""
        scores = self.bm25.get_scores(tokenize(query))
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [{**self.corpus[i], "score": float(scores[i])} for i in top_indices]


class TFIDFRetriever:
    """Baseline retrieval model using TF-IDF and cosine similarity."""

    def __init__(self, corpus: list[dict]):
        self.corpus = corpus
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        self.corpus_matrix = self.vectorizer.fit_transform([doc["text"] for doc in corpus])

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Return the top-k documents with their cosine similarity 'score'."""
        similarities = cosine_similarity(self.vectorizer.transform([query]), self.corpus_matrix).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [{**self.corpus[i], "score": float(similarities[i])} for i in top_indices]

#=====================================================================================================================
#=====================================================================================================================
#=====================================================================================================================


def build_faiss_index(embeddings, config):
    dimension = embeddings.shape[1]

    if config["index_type"] == "IndexFlatIP":
        index = faiss.IndexFlatIP(dimension)
    else:
        raise ValueError(
            f"Unsupported index type: {config['index_type']}"
        )

    index.add(
        np.asarray(embeddings, dtype="float32")
    )

    return index


def save_faiss_index(index, chunks, output_dir):
    """Save FAISS index and chunk metadata."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(
        index,
        str(output_dir / "faiss.index")
    )

    with open(
        output_dir / "chunks.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            chunks,
            f,
            ensure_ascii=False,
            indent=2
        )


def search_similar_chunks(
    query,
    index,
    chunks,
    model,
    top_k=5
):
    """Find the most similar document chunks."""

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        results.append({
            "score": float(score),
            "chunk": chunks[idx]
        })

    return results
