import pymupdf

pdf_path = "data/seminar9.pdf"

doc = pymupdf.open(pdf_path)

full_text = ""

for page in doc:
    full_text += page.get_text() + "\n"

doc.close()

words = full_text.split()

chunk_size = 50
overlap = 10

chunks = []

start = 0

while start < len(words):
    end = start + chunk_size

    chunk = " ".join(words[start:end])
    chunks.append(chunk)

    start += chunk_size - overlap

for number, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {number} ---")
    print(chunk)