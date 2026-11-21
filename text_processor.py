
from pathlib import Path
import fitz


class TextProcessor:
    """Extract text from PDF and TXT files."""

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

            return [{
                "source": file_path.name,
                "page": 1,
                "text": text
            }]

        raise ValueError("Only PDF and TXT files are supported.")

    def _extract_pdf(self, file_path):
        pages = []

        with fitz.open(file_path) as document:
            for page_number, page in enumerate(document, start=1):
                text = page.get_text()

                if text.strip():
                    pages.append({
                        "source": file_path.name,
                        "page": page_number,
                        "text": text
                    })

        return pages
