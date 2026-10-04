import ollama
import faiss
import pymupdf
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import re


# ==========================================
# 1. FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Subject Guide Question Bank Assistant",
    description="RAG based AI Question Answering System",
    version="1.0"
)


# ==========================================
# 2. LOAD PDF
# ==========================================

pdf_path = "data/seminar9.pdf"

doc = pymupdf.open(pdf_path)

full_text = ""

for page in doc:
    full_text += page.get_text() + "\n"

doc.close()


# ==========================================
# 3. TEXT CHUNKING
# ==========================================

sentences = re.split(
    r'(?<=[.!?])\s+',
    full_text.strip()
)

chunk_size = 3
overlap = 1

chunks = []

start = 0

while start < len(sentences):

    end = start + chunk_size

    chunk = " ".join(
        sentences[start:end]
    ).strip()

    if chunk:
        chunks.append(chunk)

    start += chunk_size - overlap


print("Total chunks:", len(chunks))


# ==========================================
# 4. CREATE EMBEDDINGS
# ==========================================

print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embeddings = model.encode(chunks)

print("Embeddings created!")


# ==========================================
# 5. CREATE FAISS INDEX
# ==========================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("FAISS index created!")

print(
    "Number of vectors:",
    index.ntotal
)


# ==========================================
# 6. RETRIEVAL FUNCTION
# ==========================================

def retrieve(query, k=5):

    query_embedding = model.encode(
        [query]
    )

    distances, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for distance, idx in zip(
        distances[0],
        indices[0]
    ):

        if idx < 0:
            continue

        results.append(
            chunks[idx]
        )

    return results


# ==========================================
# 7. BUILD CONTEXT
# ==========================================

def build_context(results):

    context = ""

    for result in results:

        context += result
        context += "\n\n"

    return context


# ==========================================
# 8. QUESTION MODEL
# ==========================================

class Question(BaseModel):

    question: str


# ==========================================
# 9. API ENDPOINT
# ==========================================

@app.post("/ask")
def ask_question(data: Question):

    query = data.question

    results = retrieve(
        query,
        k=5
    )

    context = build_context(
        results
    )

    prompt = f"""
You are a helpful academic assistant.

Answer the student's question using the
provided document context.

Use ONLY the information available in the
context.

If the answer is not available in the
context, say:

"I could not find the answer in the uploaded document."

Give a clear and simple answer.

Context:
{context}

Question:
{query}

Answer:
"""

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response[
        "message"
    ][
        "content"
    ]

    return {
        "question": query,
        "answer": answer,
        "sources": results
    }


# ==========================================
# 10. HOME PAGE
# ==========================================

@app.get("/")
def home():

    return FileResponse(
        "static/index.html"
    )