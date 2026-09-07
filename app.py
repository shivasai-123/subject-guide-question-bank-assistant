import os
import shutil
import ollama

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel

from document_processor import process_pdf
from rag_engine import RAGEngine
from learning_tools import LearningTools


# ==========================================
# 1. FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Subject Guide & Question Bank AI Assistant",
    description="Multi-document RAG academic learning assistant",
    version="3.0"
)


# ==========================================
# 2. INITIALIZE ENGINES
# ==========================================

rag = RAGEngine()

learning = LearningTools(
    model="llama3.2:3b"
)


# ==========================================
# 3. DOCUMENT STORAGE
# ==========================================

DATA_FOLDER = "data"

os.makedirs(
    DATA_FOLDER,
    exist_ok=True
)


# ==========================================
# 4. LOAD EXISTING PDFS
# ==========================================

def load_existing_documents():

    document_metadata = {

        "PYTHON.pdf": {
            "subject": "Python",
            "chapter": "General"
        },

        "c_programming_basics.pdf": {
            "subject": "C-Programming",
            "chapter": "General"
        },

        "ACD UNIT- 2.pdf": {
            "subject": "Artificial Intelligence",
            "chapter": "Unit 2"
        },

        "2919.pdf": {
            "subject": "General",
            "chapter": "General"
        }
    }

    pdf_files = [
        file
        for file in os.listdir(DATA_FOLDER)
        if file.lower().endswith(".pdf")
    ]

    for filename in pdf_files:

        file_path = os.path.join(
            DATA_FOLDER,
            filename
        )

        metadata = document_metadata.get(
            filename,
            {
                "subject": "General",
                "chapter": "General"
            }
        )

        try:

            documents = process_pdf(
                file_path=file_path,
                subject=metadata["subject"],
                chapter=metadata["chapter"]
            )

            rag.add_documents(
                documents
            )

            print(
                f"Loaded: {filename} "
                f"| Subject: {metadata['subject']} "
                f"| Chapter: {metadata['chapter']}"
            )

        except Exception as e:

            print(
                f"Could not load {filename}: {e}"
            )

load_existing_documents()


# ==========================================
# 5. REQUEST MODELS
# ==========================================

class Question(BaseModel):

    question: str
    subject: str | None = None
    chapter: str | None = None


class TopicRequest(BaseModel):

    topic: str


# ==========================================
# 6. HOME PAGE
# ==========================================

@app.get("/")
def home():

    return FileResponse(
        "static/index.html"
    )


# ==========================================
# 7. HEALTH CHECK
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "running",
        "documents": rag.get_document_count(),
        "vectors": rag.get_vector_count(),
        "subjects": rag.get_subjects(),
        "model": learning.model
    }


# ==========================================
# 8. UPLOAD PDF
# ==========================================

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    subject: str = Form("General"),
    chapter: str = Form("General")
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No filename provided"
        )

    if not file.filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported currently."
        )

    file_path = os.path.join(
        DATA_FOLDER,
        file.filename
    )

    try:

        with open(file_path, "wb") as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        documents = process_pdf(
            file_path=file_path,
            subject=subject,
            chapter=chapter
        )

        rag.add_documents(
            documents
        )

        return {

            "message":
                "Document uploaded successfully",

            "filename":
                file.filename,

            "subject":
                subject,

            "chapter":
                chapter,

            "chunks_added":
                len(documents),

            "total_documents":
                rag.get_document_count(),

            "total_vectors":
                rag.get_vector_count()
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==========================================
# 9. ASK QUESTION
# ==========================================

@app.post("/ask")
def ask_question(data: Question):

    query = data.question.strip()

    if not query:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    # Clean optional filters
    subject = (
        data.subject.strip()
        if data.subject
        else None
    )

    chapter = (
        data.chapter.strip()
        if data.chapter
        else None
    )

    # Retrieve only matching subject/chapter
    results = rag.retrieve(
        query,
        k=5,
        subject=subject,
        chapter=chapter
    )

    # No relevant documents found
    if not results:

        return {

            "question": query,

            "subject": subject,

            "chapter": chapter,

            "answer":
                "No relevant study material was found for the selected subject/chapter.",

            "sources": []
        }

    context = rag.build_context(
        results
    )

    answer = learning.solve_question(
        query,
        context
    )

    return {

        "question":
            query,

        "subject":
            subject,

        "chapter":
            chapter,

        "answer":
            answer,

        "sources":
            results
    }


# ==========================================
# 10. TOPIC EXPLANATION
# ==========================================

@app.post("/explain")
def explain_topic(data: TopicRequest):

    topic = data.topic.strip()

    if not topic:

        raise HTTPException(
            status_code=400,
            detail="Topic cannot be empty."
        )

    results = rag.retrieve(
        topic,
        k=5
    )

    context = rag.build_context(
        results
    )

    answer = learning.explain_topic(
        topic,
        context
    )

    return {

        "topic":
            topic,

        "answer":
            answer,

        "sources":
            results
    }


# ==========================================
# 11. CONTENT SYNTHESIS
# ==========================================

@app.post("/synthesize")
def synthesize_content(data: TopicRequest):

    topic = data.topic.strip()

    if not topic:

        raise HTTPException(
            status_code=400,
            detail="Topic cannot be empty."
        )

    results = rag.retrieve(
        topic,
        k=5
    )

    context = rag.build_context(
        results
    )

    answer = learning.synthesize_content(
        topic,
        context
    )

    return {

        "topic":
            topic,

        "answer":
            answer,

        "sources":
            results
    }


# ==========================================
# 12. LEARNING PROGRESSION
# ==========================================

@app.post("/progression")
def progression(data: TopicRequest):

    topic = data.topic.strip()

    if not topic:

        raise HTTPException(
            status_code=400,
            detail="Topic cannot be empty."
        )

    results = rag.retrieve(
        topic,
        k=5
    )

    context = rag.build_context(
        results
    )

    answer = learning.learning_progression(
        topic,
        context
    )

    return {

        "topic":
            topic,

        "answer":
            answer,

        "sources":
            results
    }


# ==========================================
# 13. LEARNING HISTORY
# ==========================================

@app.get("/history")
def get_history():

    return {
        "history":
            learning.get_history()
    }


# ==========================================
# 14. SUBJECTS
# ==========================================

@app.get("/subjects")
def get_subjects():

    return {
        "subjects":
            rag.get_subjects()
    }


# ==========================================
# 15. CHAPTERS
# ==========================================

@app.get("/chapters")
def get_chapters(
    subject: str | None = None
):

    return {

        "subject":
            subject,

        "chapters":
            rag.get_chapters(
                subject
            )
    }