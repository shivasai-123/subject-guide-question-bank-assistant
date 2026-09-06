import os
import re
import pymupdf


# ==========================================
# 1. EXTRACT TEXT FROM PDF
# ==========================================

def extract_text_from_pdf(file_path):
    """
    Extract all text from a PDF.
    """

    doc = pymupdf.open(file_path)

    full_text = ""

    for page in doc:
        full_text += page.get_text() + "\n"

    doc.close()

    return full_text


# ==========================================
# 2. CHUNK TEXT
# ==========================================

def chunk_text(text, chunk_size=3, overlap=1):
    """
    Split text into overlapping sentence chunks.
    """

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text.strip()
    )

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

    return chunks


# ==========================================
# 3. PROCESS ONE PDF
# ==========================================

def process_pdf(
    file_path,
    subject="General",
    chapter="General"
):
    """
    Process one PDF and attach metadata.
    """

    text = extract_text_from_pdf(file_path)

    chunks = chunk_text(text)

    filename = os.path.basename(file_path)

    documents = []

    for chunk in chunks:

        documents.append({
            "text": chunk,
            "metadata": {
                "filename": filename,
                "subject": subject,
                "chapter": chapter
            }
        })

    return documents


# ==========================================
# 4. PROCESS MULTIPLE PDFs
# ==========================================

def process_multiple_pdfs(documents):
    """
    Process multiple PDFs.

    Expected input:

    [
        {
            "file_path": "data/dbms.pdf",
            "subject": "DBMS",
            "chapter": "Normalization"
        },
        {
            "file_path": "data/os.pdf",
            "subject": "Operating Systems",
            "chapter": "Processes"
        }
    ]
    """

    all_documents = []

    for document in documents:

        processed = process_pdf(
            file_path=document["file_path"],
            subject=document.get(
                "subject",
                "General"
            ),
            chapter=document.get(
                "chapter",
                "General"
            )
        )

        all_documents.extend(
            processed
        )

    return all_documents