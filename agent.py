
class NSBMFaqAgent:
    """Agent that retrieves and validates relevant FAQ evidence."""

    def __init__(self, vector_store, answer_generator, top_k=3,
                 max_distance=1.2):
        self.vector_store = vector_store
        self.answer_generator = answer_generator
        self.top_k = top_k
        self.max_distance = max_distance

    def run(self, question):
        """Process a question and generate an evidence-based answer."""

        # 1. Validate user input
        if not isinstance(question, str):
            return self._error_response(
                "",
                "Invalid question format.",
                "Please enter your question as text."
            )

        observation = question.strip()

        if not observation:
            return self._error_response(
                observation,
                "No question was provided.",
                "Please enter a question."
            )

        # 2. Retrieve FAQ evidence
        try:
            results = self.vector_store.search(
                observation,
                top_k=self.top_k
            )
        except Exception as error:
            return self._error_response(
                observation,
                f"FAQ retrieval failed: {type(error).__name__}.",
                "Sorry, I could not search the NSBM FAQ documents. "
                "Please try again."
            )

        # 3. Validate retrieved results
        if not results:
            return self._no_evidence_response(observation)

        relevant_results = []

        for result in results:
            if not isinstance(result, dict):
                continue

            distance = result.get("distance")

            try:
                distance = float(distance)
            except (TypeError, ValueError):
                continue

            if distance < 0 or distance > self.max_distance:
                continue

            if not result.get("text"):
                continue

            relevant_results.append(result)

        if not relevant_results:
            return self._no_evidence_response(observation)

        # 4. Generate answer using relevant evidence
        try:
            answer = self.answer_generator.generate(
                observation,
                relevant_results
            )

            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Answer generator returned an empty answer.")

        except Exception as error:
            return self._error_response(
                observation,
                f"Answer generation failed: {type(error).__name__}.",
                "Sorry, I could not generate an answer. "
                "Please try again."
            )

        # 5. Return agent response
        return {
            "question": observation,
            "reasoning": (
                "Relevant FAQ evidence was retrieved and validated."
            ),
            "action": "Search FAQ Vector Store",
            "results": relevant_results,
            "answer": answer.strip(),
            "status": "success"
        }

    def _no_evidence_response(self, question):
        """Return a response when no relevant FAQ evidence is available."""

        return {
            "question": question,
            "reasoning": (
                "No sufficiently relevant FAQ evidence was found."
            ),
            "action": "Search FAQ Vector Store",
            "results": [],
            "answer": (
                "I could not find relevant information in the NSBM "
                "FAQ documents. Please contact NSBM for confirmation."
            ),
            "status": "no_evidence"
        }

    def _error_response(self, question, reasoning, answer):
        """Return a consistent error response."""

        return {
            "question": question,
            "reasoning": reasoning,
            "action": "Agent execution",
            "results": [],
            "answer": answer,
            "status": "error"
        }