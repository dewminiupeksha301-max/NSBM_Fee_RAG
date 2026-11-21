## Evaluation Results

The evaluation script tests 10 questions using TF-IDF, ChromaDB, and the RAG pipeline.

The evaluation results are saved in `evaluation_results.csv`.

The evaluation uses simple text-overlap-based metrics:
- Context Relevance
- Faithfulness
- Answer Relevance

These scores provide a basic comparison of retrieval and answer quality. They are not advanced semantic evaluation metrics.

## Output Files

- `evaluation_results.csv` - Contains evaluation results for the tested questions.
- `evaluation_dataset.json` - Contains the test questions and expected answers.

## Limitations

The system uses a small FAQ dataset and simple text-overlap-based evaluation metrics. Its answers depend on the information available in the source documents.