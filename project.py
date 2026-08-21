import faiss
import pymupdf
from sentence_transformers import SentenceTransformer


# -----------------------------
# 1. PDF TEXT EXTRACTION
# -----------------------------

pdf_path = "data/seminar9.pdf"

doc = pymupdf.open(pdf_path)

full_text = ""

for page in doc:
    full_text += page.get_text() + "\n"

doc.close()


# -----------------------------
# 2. TEXT CHUNKING
# -----------------------------

import re

# Split document into sentences
sentences = re.split(r'(?<=[.!?])\s+', full_text.strip())

chunk_size = 3
overlap = 1

chunks = []

start = 0

while start < len(sentences):

    end = start + chunk_size

    chunk = " ".join(sentences[start:end]).strip()

    if chunk:
        chunks.append(chunk)

    start += chunk_size - overlap

# -----------------------------
# 3. DISPLAY CHUNKS
# -----------------------------

print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {i} ---")
    print(chunk)

# -----------------------------
# 4. CREATE EMBEDDINGS
# -----------------------------

print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

print("\nEmbeddings created!")
print("Number of chunks:", len(chunks))
print("Embedding dimensions:", embeddings.shape)


# -----------------------------
# 5. CREATE FAISS INDEX
# -----------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("\nFAISS index created!")
print("Number of vectors in index:", index.ntotal)

# -----------------------------
# 6. RETRIEVAL FUNCTION
# -----------------------------

def retrieve(query, k=3):

    # Convert the question into an embedding
    query_embedding = model.encode([query])

    # Search FAISS
    distances, indices = index.search(query_embedding, k)

    results = []

    for distance, idx in zip(distances[0], indices[0]):

        results.append({
            "chunk": chunks[idx],
            "distance": float(distance),
            "index": int(idx)
        })

    return results


# -----------------------------
# 7. TEST RETRIEVAL
# -----------------------------

query = "What is minimum support?"

results = retrieve(query)

print("\nQuery:", query)

print("\nRetrieved results:")

for rank, result in enumerate(results, start=1):

    print(f"\n--- Result {rank} ---")
    print("Chunk index:", result["index"])
    print("Distance:", result["distance"])
    print("Text:")
    print(result["chunk"])

# -----------------------------
# 8. BUILD CONTEXT
# -----------------------------

def build_context(results):

    context = ""

    for result in results:
        context += result["chunk"] + "\n\n"

    return context

# -----------------------------
# 9. TEST CONTEXT
# -----------------------------

query = "What is minimum support?"

results = retrieve(query)

context = build_context(results)

print("\n==============================")
print("QUERY")
print("==============================")

print(query)

print("\n==============================")
print("RETRIEVED CONTEXT")
print("==============================")

print(context)