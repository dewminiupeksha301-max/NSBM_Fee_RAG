
class AnswerGenerator:
    """Generate answers using retrieved FAQ evidence."""

    def generate(self, question, results):
        if not results:
            return (
                "I could not find supporting information "
                "in the NSBM FAQ documents. Please contact "
                "NSBM for confirmation."
            )

        evidence = []

        for result in results:
            text = result.get("text", "").strip()

            if not text:
                continue

            # Remove the FAQ question heading
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

            cleaned_text = " ".join(answer_lines).strip()

            if cleaned_text:
                evidence.append(cleaned_text)

        if not evidence:
            return (
                "I could not find supporting information "
                "in the NSBM FAQ documents. Please contact "
                "NSBM for confirmation."
            )

        return (
            "Based on the available NSBM FAQ information:\n"
            + "\n".join(evidence)
        )
