
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Retriever:
    """Find and rank relevant FAQ text chunks using TF-IDF."""

    def __init__(self, min_score=0.0):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True
        )

        self.chunks = []
        self.chunk_vectors = None
        self.min_score = min_score

    def fit(self, chunks):
        """Build TF-IDF vectors from FAQ document chunks."""

        if not chunks:
            raise ValueError("No text chunks were provided.")

        valid_chunks = []
        seen_chunks = set()

        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue

            text = chunk.get("text", "")

            if not isinstance(text, str) or not text.strip():
                continue

            normalized = text.strip().casefold()

            if normalized in seen_chunks:
                continue

            seen_chunks.add(normalized)
            valid_chunks.append(chunk)

        if not valid_chunks:
            raise ValueError("No valid text chunks were found.")

        self.chunks = valid_chunks

        texts = [chunk["text"] for chunk in self.chunks]

        try:
            self.chunk_vectors = self.vectorizer.fit_transform(texts)
        except ValueError as error:
            raise ValueError(
                "Unable to build TF-IDF vectors. "
                "Check whether the FAQ documents contain searchable text."
            ) from error

        print("Number of chunks:", len(self.chunks))
        print(
            "TF-IDF vocabulary size:",
            len(self.vectorizer.vocabulary_)
        )

    def search(self, question, top_k=3):
        """Return the most relevant FAQ chunks."""

        if self.chunk_vectors is None:
            raise RuntimeError("Call fit() before search().")

        if not isinstance(question, str) or not question.strip():
            return []

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        question = question.strip()

        question_vector = self.vectorizer.transform([question])

        if question_vector.nnz == 0:
            return []

        scores = cosine_similarity(
            question_vector,
            self.chunk_vectors
        )[0]

        ranked_indices = scores.argsort()[::-1]

        results = []

        for index in ranked_indices:

            score = float(scores[index])

            if score <= self.min_score:
                continue

            chunk = self.chunks[index]

            results.append({
                "source": chunk.get("source", "Unknown"),
                "page": chunk.get("page", "Unknown"),
                "text": chunk["text"],
                "score": round(score, 4)
            })

            if len(results) >= top_k:
                break

        return results