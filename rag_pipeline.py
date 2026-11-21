
from pathlib import Path

from vector_store import VectorStore
from answer_generator import AnswerGenerator
from agent import NSBMFaqAgent
from retriever import Retriever
from text_processor import TextProcessor
from text_chunker import TextChunker


class RAGPipeline:
    """Connect TF-IDF, ChromaDB, agent, and answer generator."""

    def __init__(self):
        self.vector_store = VectorStore()
        self.answer_generator = AnswerGenerator()

        # Load FAQ documents for TF-IDF search
        base_folder = Path(__file__).parent
        data_folder = base_folder / "data"

        pdf_file = data_folder / "nsbm_faq.pdf"
        txt_file = data_folder / "nsbm_faq.txt"

        if pdf_file.exists():
            selected_file = pdf_file
        elif txt_file.exists():
            selected_file = txt_file
        else:
            raise FileNotFoundError(
                "No NSBM FAQ PDF or TXT file found in the data folder."
            )

        processor = TextProcessor()
        pages = processor.extract_text(selected_file)

        chunker = TextChunker()
        chunks = chunker.split_pages(pages)

        self.retriever = Retriever()
        self.retriever.fit(chunks)

        self.agent = NSBMFaqAgent(
            self.vector_store,
            self.answer_generator
        )

    
    def ask(self, question):
        # Use the agent's relevance check
        response = self.agent.run(question)

        # If the agent finds no relevant FAQ, return fallback
        if not response["results"]:
            return response

        # Prefer the TF-IDF result when it has a positive score
        tfidf_results = self.retriever.search(
            question,
            top_k=1
        )

        if tfidf_results and tfidf_results[0]["score"] > 0:
            best = tfidf_results[0]

            response["results"] = [{
                "text": best["text"],
                "source": best["source"],
                "page": best["page"],
                "distance": 0.0
            }]

            response["answer"] = self.answer_generator.generate(
                question,
                response["results"]
            )

        return response
