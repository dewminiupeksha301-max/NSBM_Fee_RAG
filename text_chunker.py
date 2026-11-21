
import re


class TextChunker:
    """Split FAQ text into question-based chunks."""

    def split_pages(self, pages):
        chunks = []

        for page in pages:
            text = page["text"]

            # Find FAQ question numbers, e.g. 1. or 6.
            parts = re.split(
                r"(?m)(?=^\s*\d+\.\s+)",
                text
            )

            for part in parts:
                part = part.strip()

                if len(part) < 15:
                    continue

                chunks.append({
                    "source": page["source"],
                    "page": page["page"],
                    "text": part
                })

        return chunks
