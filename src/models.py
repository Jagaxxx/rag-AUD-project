# Manages the vector database (FAISS) for similarity search and interfaces with the LLM reader to generate answers based on retrieved contexts.

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class BM25Retriever:
    """
    Baseline retrieval model using the BM25 algorithm.
    """
    def __init__(self, corpus: list[dict]):
        self.corpus = corpus
        tokenized_corpus = [doc["text"].lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Retrieves the top-k most relevant paragraphs based on BM25 scores.
        """
        tokenized_query = query.lower().split()
        return self.bm25.get_top_n(tokenized_query, self.corpus, n=top_k)


class TFIDFRetriever:
    """
    Baseline retrieval model using TF-IDF and Cosine Similarity.
    """
    def __init__(self, corpus: list[dict]):
        self.corpus = corpus
        self.texts = [doc["text"] for doc in corpus]
    
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.corpus_matrix = self.vectorizer.fit_transform(self.texts)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Retrieves the top-k most relevant paragraphs based on cosine similarity of TF-IDF vectors.
        """
        query_matrix = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_matrix, self.corpus_matrix).flatten()
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doc = self.corpus[idx].copy()
            doc["score"] = float(similarities[idx])
            results.append(doc)
            
        return results