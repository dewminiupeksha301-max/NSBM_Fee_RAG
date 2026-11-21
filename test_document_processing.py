from text_processor import TextProcessor
from text_chunker import TextChunker

# Initialize classes
processor = TextProcessor()
chunker = TextChunker(chunk_size=500, overlap=50)

# Extract text from actual PDF
print("Testing NSBM PDF Processing...")

pages = processor.extract_text("data/nsbm_faq.pdf")

print("Total Pages Extracted:", len(pages))

# Convert extracted pages into chunks
chunks = chunker.split_pages(pages)

print("Total Chunks Generated:", len(chunks))

# Display first 3 chunks
for index, chunk in enumerate(chunks[:3], start=1):

    print(f"\nChunk {index}")
    print("Source:", chunk["source"])
    print("Page:", chunk["page"])
    print("Text:", chunk["text"][:300])

# Validate results
assert len(pages) > 0, "No PDF pages extracted!"
assert len(chunks) > 0, "No text chunks generated!"

print("\nPDF Processing and Chunking Successful!")