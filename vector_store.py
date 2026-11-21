
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv

load_dotenv()


class VectorStore:
    """Store and search FAQ chunks persistently."""

    def __init__(self, persist_directory=None):
        if persist_directory is None:
            default_path = Path(__file__).parent / "chroma_db"
            persist_directory = os.environ.get(
                "CHROMA_DB_PATH",
                str(default_path)
            )

        persist_directory = Path(persist_directory)

        if not persist_directory.is_absolute():
            persist_directory = Path(__file__).parent / persist_directory

        self.client = chromadb.PersistentClient(
            path=str(persist_directory)
        )

        self.collection = self.client.get_or_create_collection(
            name="nsbm_faq"
        )

    def add_chunks(self, chunks):
        if not chunks:
            return

        ids = [f"chunk_{i}" for i in range(len(chunks))]
        documents = [chunk["text"] for chunk in chunks]

        metadatas = [
            {
                "source": str(chunk["source"]),
                "page": int(chunk["page"])
            }
            for chunk in chunks
        ]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    def search(self, question, top_k=3):
        count = self.collection.count()

        if count == 0:
            return []

        results = self.collection.query(
            query_texts=[question],
            n_results=count
        )

        matches = []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for i, document in enumerate(documents):
            matches.append({
                "text": document,
                "source": metadatas[i]["source"],
                "page": metadatas[i]["page"],
                "distance": distances[i]
            })

        # Sort by distance: smaller means more similar.
        matches.sort(key=lambda item: item["distance"])

        return matches[:top_k]
