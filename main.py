
from pathlib import Path

from text_processor import TextProcessor
from text_chunker import TextChunker
from retriever import Retriever
from vector_store import VectorStore
from rag_pipeline import RAGPipeline


def main():
    data_folder = Path(__file__).parent / "data"
    pdf_file = data_folder / "nsbm_faq.pdf"
    txt_file = data_folder / "nsbm_faq.txt"

    if pdf_file.exists():
        selected_file = pdf_file
    elif txt_file.exists():
        selected_file = txt_file
    else:
        print("ERROR: No PDF or TXT file found.")
        return

    processor = TextProcessor()
    pages = processor.extract_text(selected_file)

    if not pages:
        print("ERROR: No readable text found.")
        return

    chunker = TextChunker()
    chunks = chunker.split_pages(pages)

    if not chunks:
        print("ERROR: No text chunks created.")
        return

    print("Text extracted successfully.")
    print("Chunks created:", len(chunks))

    # Save FAQ chunks in persistent storage
    vector_store = VectorStore()
    vector_store.add_chunks(chunks)

    # Create the RAG pipeline
    rag_pipeline = RAGPipeline()

    # Keep TF-IDF available for comparison
    retriever = Retriever()
    retriever.fit(chunks)

    print("\nNSBM Fee Payment FAQ Assistant")
    print("Type 'vector' for RAG / Vector Store search.")
    print("Type 'exit' to stop.")

    while True:
        question = input("\nEnter your question: ").strip()

        if question.lower() == "exit":
            print("Program ended.")
            break

        if not question:
            print("Please enter a question.")
            continue

        if question.lower() == "vector":
            question = input("Enter your Vector Store question: ").strip()

            if not question:
                print("Please enter a question.")
                continue

            response = rag_pipeline.ask(question)

            print("\nObservation:", response["question"])
            print("Reasoning:", response["reasoning"])
            print("Action: Search FAQ Vector Store")
            print("Persistence: FAQ data stored in ChromaDB")
            print("\nAnswer:")
            print(response["answer"])

            if response["results"]:
                print("\nSupporting information:")

                for result in response["results"]:
                    print("\nSource:", result["source"])
                    print("Page:", result["page"])
                    print("Distance:", round(result["distance"], 4))
                    print("Text:", result["text"])

        else:
            results = retriever.search(question, top_k=3)

            print("\nTF-IDF search results:")

            if not results:
                print("No relevant information found.")
                continue

            for result in results:
                print("\nSource:", result["source"])
                print("Page:", result["page"])
                print("Similarity:", round(result["score"], 4))
                print("Text:", result["text"])


if __name__ == "__main__":
    main()
