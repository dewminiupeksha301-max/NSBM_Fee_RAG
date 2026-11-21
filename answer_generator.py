
class AnswerGenerator:
    """Generate clear answers using retrieved FAQ evidence."""

    FALLBACK_MESSAGE = (
        "I could not find supporting information "
        "in the NSBM FAQ documents. Please contact "
        "NSBM for confirmation."
    )

    def generate(self, question, results):
        """Generate an answer from relevant FAQ search results."""

        # 1. Validate question
        if not isinstance(question, str) or not question.strip():
            return "Please enter a valid question."

        # 2. Validate retrieved results
        if not results or not isinstance(results, list):
            return self.FALLBACK_MESSAGE

        evidence = []
        seen_evidence = set()

        # 3. Process retrieved FAQ evidence
        for result in results:

            if not isinstance(result, dict):
                continue

            text = result.get("text", "")

            if not isinstance(text, str) or not text.strip():
                continue

            cleaned_text = self._clean_evidence(text)

            if not cleaned_text:
                continue

            # 4. Remove duplicate evidence
            normalized_text = cleaned_text.casefold()

            if normalized_text not in seen_evidence:
                evidence.append(cleaned_text)
                seen_evidence.add(normalized_text)

        # 5. Return fallback if no valid evidence exists
        if not evidence:
            return self.FALLBACK_MESSAGE

        # 6. Generate final answer
        return (
            "Based on the available NSBM FAQ information:\n"
            + "\n".join(evidence)
        )

    def _clean_evidence(self, text):
        """Remove FAQ headings and unrelated page headings."""

        lines = text.splitlines()
        answer_lines = []

        for line in lines:
            line = line.strip()

            if not line:
                continue

            # Skip numbered FAQ headings
            if line[:1].isdigit() and ". " in line[:5]:
                continue

            # Skip unrelated page headings
            if line.upper() == "ACADEMIC PROGRAMMES":
                continue

            answer_lines.append(line)

        return " ".join(answer_lines).strip()