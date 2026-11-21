
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Retriever:
    """Find text chunks relevant to a question."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english"
        )
        self.chunks = []
        self.chunk_vectors = None

    def fit(self, chunks):
        if not chunks:
            raise ValueError("No text chunks were provided.")

        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]

        self.chunk_vectors = self.vectorizer.fit_transform(
            texts
        )

        print("Number of chunks:", len(self.chunks))
        print(
            "TF-IDF vocabulary size:",
            len(self.vectorizer.vocabulary_)
        )

    def search(self, question, top_k=3):
        if self.chunk_vectors is None:
            raise RuntimeError("Call fit() before search().")

        question_vector = self.vectorizer.transform(
            [question]
        )

        scores = cosine_similarity(
            question_vector,
            self.chunk_vectors
        )[0]

        ranked_indices = scores.argsort()[::-1]
        results = []

        for index in ranked_indices:
            if scores[index] <= 0:
                continue

            if len(results) >= top_k:
                break

            chunk = self.chunks[index]

            results.append({
                "source": chunk["source"],
                "page": chunk["page"],
                "text": chunk["text"],
                "score": float(scores[index])
            })

        return results
