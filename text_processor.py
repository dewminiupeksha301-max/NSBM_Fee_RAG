from pathlib import Path
import fitz


class TextProcessor:
    """Extract and clean text from PDF and TXT files."""

    def extract_text(self, file_path):
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        if file_path.suffix.lower() == ".pdf":
            return self._extract_pdf(file_path)

        if file_path.suffix.lower() == ".txt":
            text = file_path.read_text(encoding="utf-8")
            text = self._clean_text(text)

            if not text:
                return []

            return [{
                "source": file_path.name,
                "page": 1,
                "text": text
            }]

        raise ValueError("Only PDF and TXT files are supported.")

    def _extract_pdf(self, file_path):
        pages = []

        try:
            with fitz.open(file_path) as document:

                for page_number, page in enumerate(document, start=1):
                    text = page.get_text()
                    text = self._clean_text(text)

                    if text:
                        pages.append({
                            "source": file_path.name,
                            "page": page_number,
                            "text": text
                        })

        except Exception as error:
            raise ValueError(
                f"Unable to process PDF: {error}"
            ) from error

        return pages

    def _clean_text(self, text):
        """Remove unnecessary spaces and blank lines."""

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        return "\n".join(lines)