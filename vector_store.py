
import os
import hashlib
from pathlib import Path

import chromadb
from dotenv import load_dotenv

load_dotenv()


class VectorStore:
    """Store and retrieve FAQ chunks using persistent ChromaDB."""

    def __init__(self, persist_directory=None):
        # 1. Configure persistent database location
        if persist_directory is None:
            default_path = Path(__file__).parent / "chroma_db"

            persist_directory = os.environ.get(
                "CHROMA_DB_PATH",
                str(default_path)
            )

        persist_directory = Path(persist_directory)

        if not persist_directory.is_absolute():
            persist_directory = Path(__file__).parent / persist_directory

        persist_directory.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(persist_directory)
        )

        # 2. Create or load FAQ collection
        self.collection = self.client.get_or_create_collection(
            name="nsbm_faq",
            metadata={"description": "NSBM FAQ document collection"}
        )

    def add_chunks(self, chunks):
        """Validate and store FAQ chunks."""

        if not chunks:
            print("No chunks provided for storage.")
            return

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue

            text = chunk.get("text", "")

            if not isinstance(text, str) or not text.strip():
                continue

            source = str(chunk.get("source", "Unknown"))
            page = chunk.get("page", 0)

            try:
                page = int(page)
            except (TypeError, ValueError):
                page = 0

            # Generate a stable ID for each chunk
            unique_content = (
                f"{source}|{page}|{text.strip()}"
            )

            chunk_id = hashlib.sha256(
                unique_content.encode("utf-8")
            ).hexdigest()

            ids.append(chunk_id)
            documents.append(text.strip())

            metadatas.append({
                "source": source,
                "page": page
            })

        if not documents:
            print("No valid chunks found.")
            return

        # 3. Store documents in ChromaDB
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

        print("Successfully stored chunks:", len(documents))

    def search(self, question, top_k=3):
        """Search FAQ documents using vector similarity."""

        # 1. Validate question
        if not isinstance(question, str) or not question.strip():
            return []

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        # 2. Check database collection
        count = self.collection.count()

        if count == 0:
            return []

        # 3. Perform vector search
        try:
            results = self.collection.query(
                query_texts=[question.strip()],
                n_results=min(top_k, count),
                include=["documents", "metadatas", "distances"]
            )

        except Exception as error:
            raise RuntimeError(
                "FAQ vector search failed."
            ) from error

        # 4. Extract search results
        matches = []

        documents = results.get("documents") or [[]]
        metadatas = results.get("metadatas") or [[]]
        distances = results.get("distances") or [[]]

        if not documents or not documents[0]:
            return []

        for document, metadata, distance in zip(
            documents[0],
            metadatas[0],
            distances[0]
        ):
            if not document:
                continue

            metadata = metadata or {}

            matches.append({
                "text": document,
                "source": metadata.get("source", "Unknown"),
                "page": metadata.get("page", 0),
                "distance": float(distance)
            })

        # 5. Return closest matches first
        matches.sort(
            key=lambda item: item["distance"]
        )

        return matches

    def get_collection_count(self):
        """Return the number of stored FAQ chunks."""

        return self.collection.count()