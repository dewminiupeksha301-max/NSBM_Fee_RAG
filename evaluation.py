import json
import csv
from pathlib import Path

from text_processor import TextProcessor
from text_chunker import TextChunker
from retriever import Retriever
from vector_store import VectorStore
from rag_pipeline import RAGPipeline


def calculate_overlap(text1, text2):
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1:
        return 0.0

    return len(words1 & words2) / len(words1)


def evaluate_retriever(dataset, search_function, method_name):
    context_scores = []
    faithfulness_scores = []
    answer_scores = []
    records = []

    print(f"\n{'=' * 50}")
    print(f"EVALUATION: {method_name}")
    print("=" * 50)

    for item in dataset:
        question = item["question"]
        expected_answer = item["answer"]
        expected_context = item["context"]

        results = search_function(question, top_k=1)

        if not results:
            retrieved_text = ""
            context_score = 0.0
            faithfulness_score = 0.0
        else:
            retrieved_text = results[0]["text"]

            context_score = calculate_overlap(
                expected_context, retrieved_text
            )

            faithfulness_score = calculate_overlap(
                expected_answer, retrieved_text
            )

        answer_score = calculate_overlap(question, expected_answer)

        context_scores.append(context_score)
        faithfulness_scores.append(faithfulness_score)
        answer_scores.append(answer_score)

        records.append({
            "method": method_name,
            "record_type": "Question",
            "question": question,
            "retrieved_text": retrieved_text,
            "generated_answer": "",
            "expected_answer": expected_answer,
            "context_relevance": round(context_score, 3),
            "faithfulness": round(faithfulness_score, 3),
            "answer_relevance": round(answer_score, 3)
        })

        print("\nQuestion:", question)
        print("Retrieved text:", retrieved_text)
        print("Context Relevance:", round(context_score, 3))
        print("Groundedness/Faithfulness:", round(faithfulness_score, 3))
        print("Answer Relevance:", round(answer_score, 3))

    count = len(dataset)

    if count == 0:
        print("Evaluation dataset is empty.")
        return records

    average_context = sum(context_scores) / count
    average_faithfulness = sum(faithfulness_scores) / count
    average_answer = sum(answer_scores) / count

    print("\nAVERAGE SCORES")
    print("Context Relevance:", round(average_context, 3))
    print("Groundedness/Faithfulness:", round(average_faithfulness, 3))
    print("Answer Relevance:", round(average_answer, 3))

    records.append({
        "method": method_name,
        "record_type": "Average",
        "question": "",
        "retrieved_text": "",
        "generated_answer": "",
        "expected_answer": "",
        "context_relevance": round(average_context, 3),
        "faithfulness": round(average_faithfulness, 3),
        "answer_relevance": round(average_answer, 3)
    })

    return records


def evaluate_rag_answers(dataset, pipeline):
    records = []

    print("\n" + "=" * 50)
    print("RAG PIPELINE ANSWER TEST")
    print("=" * 50)

    for item in dataset:
        response = pipeline.ask(item["question"])

        generated_answer = response["answer"]
        expected_answer = item["answer"]
        expected_context = item["context"]

        context_score = calculate_overlap(
            expected_context, generated_answer
        )

        faithfulness_score = calculate_overlap(
            expected_answer, generated_answer
        )

        answer_score = calculate_overlap(
            item["question"], generated_answer
        )

        records.append({
            "method": "RAG Pipeline",
            "record_type": "Question",
            "question": item["question"],
            "retrieved_text": "",
            "generated_answer": generated_answer,
            "expected_answer": expected_answer,
            "context_relevance": round(context_score, 3),
            "faithfulness": round(faithfulness_score, 3),
            "answer_relevance": round(answer_score, 3)
        })

        print("\nQuestion:", item["question"])
        print("Generated answer:", generated_answer)
        print("Expected answer:", expected_answer)
        print("Context Relevance:", round(context_score, 3))
        print("Faithfulness:", round(faithfulness_score, 3))
        print("Answer Relevance:", round(answer_score, 3))

    return records


def main():
    base_folder = Path(__file__).parent
    data_folder = base_folder / "data"

    dataset_file = base_folder / "evaluation_dataset.json"
    pdf_file = data_folder / "nsbm_faq.pdf"
    txt_file = data_folder / "nsbm_faq.txt"
    output_file = base_folder / "evaluation_results.csv"

    if not dataset_file.exists():
        print("ERROR: evaluation_dataset.json not found.")
        return

    if pdf_file.exists():
        selected_file = pdf_file
    elif txt_file.exists():
        selected_file = txt_file
    else:
        print("ERROR: No PDF or TXT file found.")
        return

    with open(dataset_file, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not dataset:
        print("ERROR: Evaluation dataset is empty.")
        return

    processor = TextProcessor()
    pages = processor.extract_text(selected_file)

    chunker = TextChunker()
    chunks = chunker.split_pages(pages)

    if not chunks:
        print("ERROR: No text chunks created.")
        return

    all_records = []

    # 1. Evaluate TF-IDF retrieval
    retriever = Retriever()
    retriever.fit(chunks)

    all_records.extend(
        evaluate_retriever(
            dataset,
            retriever.search,
            "TF-IDF"
        )
    )

    # 2. Evaluate ChromaDB retrieval
    vector_store = VectorStore()
    vector_store.add_chunks(chunks)

    all_records.extend(
        evaluate_retriever(
            dataset,
            vector_store.search,
            "ChromaDB Vector Store"
        )
    )

    # 3. Evaluate RAG generated answers
    pipeline = RAGPipeline()

    all_records.extend(
        evaluate_rag_answers(dataset, pipeline)
    )

    # 4. Save all evaluation results to CSV
    fieldnames = [
        "method",
        "record_type",
        "question",
        "retrieved_text",
        "generated_answer",
        "expected_answer",
        "context_relevance",
        "faithfulness",
        "answer_relevance"
    ]

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_records)

    print("\n" + "=" * 50)
    print("EVALUATION COMPLETED")
    print("=" * 50)
    print("Total test questions:", len(dataset))
    print("Results saved to:", output_file)


if __name__ == "__main__":
    main()