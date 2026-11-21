from text_processor import TextProcessor

processor = TextProcessor()

pdf_path = "data/nsbm_faq.pdf"
txt_path = "data/nsbm_faq.txt"

# Test PDF extraction
print("Testing PDF Extraction...")

pdf_results = processor.extract_text(pdf_path)

print("PDF Pages Extracted:", len(pdf_results))

for page in pdf_results[:2]:
    print("\nPage:", page["page"])
    print("Source:", page["source"])
    print("Text:", page["text"][:200])

# Test TXT extraction
print("\nTesting TXT Extraction...")

txt_results = processor.extract_text(txt_path)

print("TXT Results:", len(txt_results))

if txt_results:
    print(txt_results[0]["text"][:200])

print("\nText Processing Test Completed!")