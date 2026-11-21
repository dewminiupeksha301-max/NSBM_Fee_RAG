import re


class TextChunker:
    """Split FAQ text into question-based chunks."""

    def __init__(self, chunk_size=500, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_pages(self, pages):
        chunks = []

        for page in pages:

            text = page.get("text", "").strip()

            if not text:
                continue

            # Split text using FAQ question numbers
            parts = re.split(
                r"(?m)(?=^\s*\d+\.\s+)",
                text
            )

            for part in parts:

                part = part.strip()

                if len(part) < 15:
                    continue

                # Split long text into smaller chunks
                text_chunks = self._split_long_text(part)

                for chunk in text_chunks:
                    chunks.append({
                        "source": page["source"],
                        "page": page["page"],
                        "text": chunk
                    })

        return chunks

    def _split_long_text(self, text):
        """Split long text with overlap."""

        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):

            end = start + self.chunk_size

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= len(text):
                break

            start = end - self.overlap

        return chunks