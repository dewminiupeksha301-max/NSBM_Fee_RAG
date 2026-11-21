from text_chunker import TextChunker

pages = [
    {
        "source": "nsbm_faq.pdf",
        "page": 1,
        "text": """
1. What is NSBM?
NSBM is a university that offers degree programmes.

2. How can students pay their fees?
Students can pay fees using available payment methods.

3. Where can students get payment information?
Students can contact the university administration.
"""
    }
]

chunker = TextChunker(chunk_size=100, overlap=20)

chunks = chunker.split_pages(pages)

print("Total Chunks:", len(chunks))

for index, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {index}")
    print("Source:", chunk["source"])
    print("Page:", chunk["page"])
    print("Text:", chunk["text"])

print("\nText Chunking Test Completed!")