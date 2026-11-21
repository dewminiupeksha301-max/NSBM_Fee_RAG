
class NSBMFaqAgent:
    """Agent that retrieves relevant FAQ evidence."""

    def __init__(self, vector_store, answer_generator):
        self.vector_store = vector_store
        self.answer_generator = answer_generator

    def run(self, question):
        observation = question.strip()

        if not observation:
            return {
                "question": observation,
                "reasoning": "No question was provided.",
                "results": [],
                "answer": "Please enter a question."
            }

        # Search the FAQ vector store
        results = self.vector_store.search(
            observation,
            top_k=1
        )

        # Reject weak matches
        
        if not results or results[0].get("distance", 999) > 1.2:

            return {
                "question": observation,
                "reasoning": "No sufficiently relevant FAQ evidence was found.",
                "results": [],
                "answer": (
                    "I could not find relevant information "
                    "in the NSBM FAQ documents. Please contact "
                    "NSBM for confirmation."
                )
            }

        best_result = results[0]

        answer = self.answer_generator.generate(
            observation,
            [best_result]
        )

        return {
            "question": observation,
            "reasoning": "Relevant FAQ evidence was retrieved.",
            "results": [best_result],
            "answer": answer
        }
