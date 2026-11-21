
from pathlib import Path

from vector_store import VectorStore
from answer_generator import AnswerGenerator
from agent import NSBMFaqAgent
from retriever import Retriever
from text_processor import TextProcessor
from text_chunker import TextChunker


class RAGPipeline:
    """Connect document processing, retrieval, agent and answer generation."""

    def __init__(self):
        self.vector_store = VectorStore()
        self.answer_generator = AnswerGenerator()

        # 1. Locate FAQ documents
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

        # 2. Extract document text
        processor = TextProcessor()
        pages = processor.extract_text(selected_file)

        if not pages:
            raise ValueError(
                "No text could be extracted from the FAQ document."
            )

        # 3. Split extracted text into chunks
        chunker = TextChunker()
        chunks = chunker.split_pages(pages)

        if not chunks:
            raise ValueError(
                "No text chunks were created from the FAQ document."
            )

        # 4. Initialize TF-IDF retriever
        self.retriever = Retriever()
        self.retriever.fit(chunks)

        # 5. Initialize FAQ agent
        self.agent = NSBMFaqAgent(
            self.vector_store,
            self.answer_generator
        )

    def ask(self, question):
        """Retrieve relevant FAQ evidence and generate an answer."""

        # 1. Validate user question
        if not isinstance(question, str) or not question.strip():
            return {
                "question": question if isinstance(question, str) else "",
                "reasoning": "Invalid or empty question.",
                "results": [],
                "answer": "Please enter a valid question.",
                "status": "error"
            }

        question = question.strip()

        # 2. Search using ChromaDB through the agent
        response = self.agent.run(question)

        if not response.get("results"):
            return response

        # 3. Search using TF-IDF
        try:
            tfidf_results = self.retriever.search(
                question,
                top_k=1
            )
        except Exception:
            tfidf_results = []

        # 4. Select the best available evidence
        if tfidf_results and tfidf_results[0].get("score", 0) > 0:

            best = tfidf_results[0]

            selected_result = {
                "text": best["text"],
                "source": best["source"],
                "page": best["page"],
                "distance": 0.0
            }

            response["results"] = [selected_result]

            # 5. Generate answer using selected evidence
            response["answer"] = self.answer_generator.generate(
                question,
                [selected_result]
            )

            response["reasoning"] = (
                "Relevant FAQ evidence was retrieved using "
                "ChromaDB and refined using TF-IDF retrieval."
            )

            response["retrieval_method"] = "ChromaDB + TF-IDF"

        else:
            response["retrieval_method"] = "ChromaDB"

        # 6. Return final response
        return response