
import json
import csv
import re
from pathlib import Path

from text_processor import TextProcessor
from text_chunker import TextChunker
from retriever import Retriever
from vector_store import VectorStore
from rag_pipeline import RAGPipeline


def tokenize(text):
    """Convert text into normalized word tokens."""

    if not isinstance(text, str):
        return set()

    return set(re.findall(r"\b[a-z0-9]+\b", text.lower()))


def calculate_overlap(reference, candidate):
    """Calculate the proportion of reference words found in candidate."""

    reference_words = tokenize(reference)
    candidate_words = tokenize(candidate)

    if not reference_words:
        return 0.0

    return len(reference_words & candidate_words) / len(reference_words)


def calculate_metrics(expected, actual):
    """Calculate precision, recall and F1-score."""

    expected_words = tokenize(expected)
    actual_words = tokenize(actual)

    if not expected_words or not actual_words:
        return 0.0, 0.0, 0.0

    common = len(expected_words & actual_words)

    precision = common / len(actual_words)
    recall = common / len(expected_words)

    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = (
            2 * precision * recall
            / (precision + recall)
        )

    return precision, recall, f1_score


def safe_average(scores):
    """Calculate average score safely."""

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


def evaluate_retriever(dataset, search_function, method_name):
    """Evaluate a document retrieval method."""

    context_scores = []
    faithfulness_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    records = []

    print("\n" + "=" * 55)
    print("EVALUATION:", method_name)
    print("=" * 55)

    for item in dataset:

        question = item["question"]
        expected_answer = item["answer"]
        expected_context = item["context"]

        try:
            results = search_function(question, top_k=1)
        except Exception as error:
            print("Retrieval error:", error)
            results = []

        retrieved_text = (
            results[0].get("text", "")
            if results else ""
        )

        context_score = calculate_overlap(
            expected_context,
            retrieved_text
        )

        faithfulness_score = calculate_overlap(
            expected_answer,
            retrieved_text
        )

        precision, recall, f1 = calculate_metrics(
            expected_context,
            retrieved_text
        )

        context_scores.append(context_score)
        faithfulness_scores.append(faithfulness_score)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        records.append({
            "method": method_name,
            "record_type": "Question",
            "question": question,
            "retrieved_text": retrieved_text,
            "generated_answer": "",
            "expected_answer": expected_answer,
            "context_relevance": round(context_score, 3),
            "faithfulness": round(faithfulness_score, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1, 3)
        })

        print("\nQuestion:", question)
        print("Context Relevance:", round(context_score, 3))
        print("Faithfulness:", round(faithfulness_score, 3))
        print("Precision:", round(precision, 3))
        print("Recall:", round(recall, 3))
        print("F1-score:", round(f1, 3))

    averages = {
        "context_relevance": safe_average(context_scores),
        "faithfulness": safe_average(faithfulness_scores),
        "precision": safe_average(precision_scores),
        "recall": safe_average(recall_scores),
        "f1_score": safe_average(f1_scores)
    }

    print("\nAVERAGE SCORES")

    for name, score in averages.items():
        print(name.replace("_", " ").title(), ":", round(score, 3))

    records.append({
        "method": method_name,
        "record_type": "Average",
        "context_relevance": round(averages["context_relevance"], 3),
        "faithfulness": round(averages["faithfulness"], 3),
        "precision": round(averages["precision"], 3),
        "recall": round(averages["recall"], 3),
        "f1_score": round(averages["f1_score"], 3)
    })

    return records


def evaluate_rag_answers(dataset, pipeline):
    """Evaluate generated answers from the RAG pipeline."""

    records = []

    print("\n" + "=" * 55)
    print("RAG PIPELINE ANSWER EVALUATION")
    print("=" * 55)

    for item in dataset:

        question = item["question"]
        expected_answer = item["answer"]
        expected_context = item["context"]

        try:
            response = pipeline.ask(question)
            generated_answer = response.get("answer", "")
        except Exception as error:
            print("RAG evaluation error:", error)
            generated_answer = ""

        context_score = calculate_overlap(
            expected_context,
            generated_answer
        )

        faithfulness_score = calculate_overlap(
            expected_answer,
            generated_answer
        )

        precision, recall, f1 = calculate_metrics(
            expected_answer,
            generated_answer
        )

        records.append({
            "method": "RAG Pipeline",
            "record_type": "Question",
            "question": question,
            "retrieved_text": "",
            "generated_answer": generated_answer,
            "expected_answer": expected_answer,
            "context_relevance": round(context_score, 3),
            "faithfulness": round(faithfulness_score, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1, 3)
        })

        print("\nQuestion:", question)
        print("Generated Answer:", generated_answer)
        print("Faithfulness:", round(faithfulness_score, 3))
        print("Precision:", round(precision, 3))
        print("Recall:", round(recall, 3))
        print("F1-score:", round(f1, 3))

    return records


def save_results(records, output_file):
    """Save evaluation records into a CSV report."""

    fieldnames = [
        "method",
        "record_type",
        "question",
        "retrieved_text",
        "generated_answer",
        "expected_answer",
        "context_relevance",
        "faithfulness",
        "precision",
        "recall",
        "f1_score"
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
        writer.writerows(records)


def main():

    base_folder = Path(__file__).parent
    data_folder = base_folder / "data"

    dataset_file = base_folder / "evaluation_dataset.json"
    pdf_file = data_folder / "nsbm_faq.pdf"
    txt_file = data_folder / "nsbm_faq.txt"

    output_file = base_folder / "evaluation_results.csv"

    # 1. Validate evaluation dataset
    if not dataset_file.exists():
        print("ERROR: Evaluation dataset not found.")
        return

    try:
        with open(dataset_file, "r", encoding="utf-8") as file:
            dataset = json.load(file)
    except (json.JSONDecodeError, OSError) as error:
        print("ERROR: Could not load evaluation dataset:", error)
        return

    if not isinstance(dataset, list) or not dataset:
        print("ERROR: Evaluation dataset is empty or invalid.")
        return

    required_fields = {"question", "answer", "context"}

    for index, item in enumerate(dataset):
        if not isinstance(item, dict) or not required_fields.issubset(item):
            print("ERROR: Invalid dataset record:", index + 1)
            return

    # 2. Locate FAQ document
    if pdf_file.exists():
        selected_file = pdf_file
    elif txt_file.exists():
        selected_file = txt_file
    else:
        print("ERROR: No FAQ PDF or TXT file found.")
        return

    try:
        # 3. Process FAQ document
        processor = TextProcessor()
        pages = processor.extract_text(selected_file)

        chunker = TextChunker()
        chunks = chunker.split_pages(pages)

        if not chunks:
            print("ERROR: No text chunks created.")
            return

        all_records = []

        # 4. Evaluate TF-IDF
        retriever = Retriever()
        retriever.fit(chunks)

        all_records.extend(
            evaluate_retriever(
                dataset,
                retriever.search,
                "TF-IDF"
            )
        )

        # 5. Evaluate ChromaDB
        vector_store = VectorStore()
        vector_store.add_chunks(chunks)

        all_records.extend(
            evaluate_retriever(
                dataset,
                vector_store.search,
                "ChromaDB Vector Store"
            )
        )

        # 6. Evaluate RAG Pipeline
        pipeline = RAGPipeline()

        all_records.extend(
            evaluate_rag_answers(
                dataset,
                pipeline
            )
        )

        # 7. Save evaluation report
        save_results(all_records, output_file)

        print("\n" + "=" * 55)
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print("=" * 55)
        print("Total test questions:", len(dataset))
        print("Total evaluation records:", len(all_records))
        print("Results saved to:", output_file)

    except Exception as error:
        print("\nEVALUATION FAILED:", error)


if __name__ == "__main__":
    main()