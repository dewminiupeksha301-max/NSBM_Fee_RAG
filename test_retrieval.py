from retriever import Retriever
from vector_store import VectorStore

# Sample FAQ chunks
chunks = [
    {
        "text": "NSBM offers degree programmes in computing and business.",
        "source": "nsbm_faq.pdf",
        "page": 1
    },
    {
        "text": "Students can contact the examination department for exam details.",
        "source": "nsbm_faq.pdf",
        "page": 2
    }
]

# Test TF-IDF Retriever
print("Testing Retriever...")
retriever = Retriever()
retriever.fit(chunks)

results = retriever.search("What degree programmes are available?", top_k=2)
print("Retriever results:", results)

# Test ChromaDB Vector Store
print("\nTesting Vector Store...")
store = VectorStore(persist_directory="test_chroma_db")
store.add_chunks(chunks)

results = store.search("What degree programmes are available?", top_k=2)
print("Vector Store results:", results)

print("\nTesting completed!")
