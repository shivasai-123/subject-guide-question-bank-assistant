import os
import shutil
import ollama

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel

from document_processor import process_file
from rag_engine import RAGEngine
from learning_tools import LearningTools


# ==========================================
# 1. FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Subject Guide & Question Bank AI Assistant",
    description="Multi-document RAG academic learning assistant",
    version="3.3"
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
# 4. DETECT CONTENT TYPE
# ==========================================

def detect_content_type(filename):

    filename_lower = filename.lower()

    question_bank_keywords = [
        "question",
        "question bank",
        "questionbank",
        "previous year",
        "previousyear",
        "pyq",
        "paper",
        "exam",
        "questionpaper",
        "question paper"
    ]

    lab_keywords = [
        "lab",
        "laboratory",
        "practical"
    ]

    textbook_keywords = [
        "textbook",
        "book"
    ]

    for keyword in question_bank_keywords:

        if keyword in filename_lower:
            return "Question Bank"

    for keyword in lab_keywords:

        if keyword in filename_lower:
            return "Lab Manual"

    for keyword in textbook_keywords:

        if keyword in filename_lower:
            return "Textbook"

    return "Notes"


# ==========================================
# 5. SUPPORTED FILE TYPES
# ==========================================

SUPPORTED_EXTENSIONS = (
    ".pdf",
    ".docx",
    ".pptx"
)


# ==========================================
# 6. LOAD EXISTING DOCUMENTS
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

        "Python Question Bank.pdf": {
            "subject": "Python",
            "chapter": "General"
        },

        "2919.pdf": {
            "subject": "General",
            "chapter": "General"
        },

        # ==================================
        # DISCRETE MATHEMATICS PPT
        # ==================================

        "DM UNIT3.pptx": {
            "subject": "DM",
            "chapter": "UNIT 3"
        }
    }


    supported_files = [
        file
        for file in os.listdir(DATA_FOLDER)
        if file.lower().endswith(
            SUPPORTED_EXTENSIONS
        )
    ]


    for filename in supported_files:

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


        content_type = detect_content_type(
            filename
        )


        try:

            documents = process_file(
                file_path=file_path,
                subject=metadata["subject"],
                chapter=metadata["chapter"]
            )


            # Add content type to every chunk
            for document in documents:

                document["metadata"][
                    "content_type"
                ] = content_type


            rag.add_documents(
                documents
            )


            print(
                f"Loaded: {filename} "
                f"| Subject: {metadata['subject']} "
                f"| Chapter: {metadata['chapter']} "
                f"| Type: {content_type} "
                f"| Chunks: {len(documents)}"
            )


        except Exception as e:

            print(
                f"Could not load {filename}: {e}"
            )


load_existing_documents()


# ==========================================
# 7. REQUEST MODELS
# ==========================================

class Question(BaseModel):

    question: str
    subject: str | None = None
    chapter: str | None = None


class TopicRequest(BaseModel):

    topic: str
    subject: str | None = None
    chapter: str | None = None


# ==========================================
# 8. HOME PAGE
# ==========================================

@app.get("/")
def home():

    return FileResponse(
        "static/index.html"
    )


# ==========================================
# 9. HEALTH CHECK
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "running",
        "documents": rag.get_document_count(),
        "vectors": rag.get_vector_count(),
        "subjects": rag.get_subjects(),
        "model": learning.model,
        "supported_formats": [
            "PDF",
            "DOCX",
            "PPTX"
        ]
    }


# ==========================================
# 10. UPLOAD DOCUMENT
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


    filename = file.filename


    if not filename.lower().endswith(
        SUPPORTED_EXTENSIONS
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Supported formats: "
                "PDF, DOCX, PPTX."
            )
        )


    file_path = os.path.join(
        DATA_FOLDER,
        filename
    )


    content_type = detect_content_type(
        filename
    )


    try:

        with open(
            file_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )


        documents = process_file(
            file_path=file_path,
            subject=subject,
            chapter=chapter
        )


        # Add content type to every chunk
        for document in documents:

            document["metadata"][
                "content_type"
            ] = content_type


        rag.add_documents(
            documents
        )


        return {

            "message":
                "Document uploaded successfully",

            "filename":
                filename,

            "subject":
                subject,

            "chapter":
                chapter,

            "content_type":
                content_type,

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
# 11. ASK QUESTION
# ==========================================

@app.post("/ask")
def ask_question(data: Question):

    query = data.question.strip()


    if not query:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )


    # --------------------------------------
    # Clean optional filters
    # --------------------------------------

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


    # ======================================
    # DETECT "LIST QUESTION BANK" QUERY
    # ======================================

    query_lower = query.lower()


    list_question_bank_query = any(
        phrase in query_lower
        for phrase in [

            "what questions are included",

            "what questions are in",

            "which questions are included",

            "which questions are in",

            "list the questions",

            "list all questions",

            "all questions in the question bank",

            "questions included in the external question bank",

            "programs included in the question bank",

            "programs are included in the question bank"
        ]
    )


    # ======================================
    # RETRIEVE MATERIAL
    # ======================================

    if list_question_bank_query:

        results = (
            rag.get_all_question_bank_documents(
                subject=subject,
                chapter=chapter
            )
        )

    else:

        results = rag.retrieve(
            query,
            k=5,
            subject=subject,
            chapter=chapter
        )


    # ======================================
    # NO RELEVANT DOCUMENTS
    # ======================================

    if not results:

        return {

            "question": query,

            "subject": subject,

            "chapter": chapter,

            "answer":
                "No relevant study material was found for the selected subject/chapter.",

            "sources": []
        }


    # ======================================
    # BUILD CONTEXT
    # ======================================

    context = rag.build_context(
        results
    )


    # ======================================
    # GENERATE ANSWER
    # ======================================

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
# 12. TOPIC EXPLANATION
# ==========================================

@app.post("/explain")
def explain_topic(data: TopicRequest):

    topic = data.topic.strip()


    if not topic:

        raise HTTPException(
            status_code=400,
            detail="Topic cannot be empty."
        )


    # --------------------------------------
    # Clean optional filters
    # --------------------------------------

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


    # --------------------------------------
    # Retrieve filtered material
    # --------------------------------------

    results = rag.retrieve(
        topic,
        k=5,
        subject=subject,
        chapter=chapter
    )


    # --------------------------------------
    # No results
    # --------------------------------------

    if not results:

        return {

            "topic": topic,

            "subject": subject,

            "chapter": chapter,

            "answer":
                "No relevant study material was found for the selected subject/chapter.",

            "sources": []
        }


    # --------------------------------------
    # Build context
    # --------------------------------------

    context = rag.build_context(
        results
    )


    # --------------------------------------
    # Generate explanation
    # --------------------------------------

    answer = learning.explain_topic(
        topic,
        context
    )


    return {

        "topic":
            topic,

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
# 13. CONTENT SYNTHESIS
# ==========================================

@app.post("/synthesize")
def synthesize_content(data: TopicRequest):

    topic = data.topic.strip()


    if not topic:

        raise HTTPException(
            status_code=400,
            detail="Topic cannot be empty."
        )


    # --------------------------------------
    # Clean optional filters
    # --------------------------------------

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


    # --------------------------------------
    # Retrieve filtered material
    # --------------------------------------

    results = rag.retrieve(
        topic,
        k=5,
        subject=subject,
        chapter=chapter
    )


    # --------------------------------------
    # No results
    # --------------------------------------

    if not results:

        return {

            "topic": topic,

            "subject": subject,

            "chapter": chapter,

            "answer":
                "No relevant study material was found for the selected subject/chapter.",

            "sources": []
        }


    # --------------------------------------
    # Build context
    # --------------------------------------

    context = rag.build_context(
        results
    )


    # --------------------------------------
    # Generate synthesis
    # --------------------------------------

    answer = learning.synthesize_content(
        topic,
        context
    )


    return {

        "topic":
            topic,

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
# 14. LEARNING PROGRESSION
# ==========================================

@app.post("/progression")
def progression(data: TopicRequest):

    topic = data.topic.strip()


    if not topic:

        raise HTTPException(
            status_code=400,
            detail="Topic cannot be empty."
        )


    # --------------------------------------
    # Clean optional filters
    # --------------------------------------

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


    # --------------------------------------
    # Retrieve filtered material
    # --------------------------------------

    results = rag.retrieve(
        topic,
        k=5,
        subject=subject,
        chapter=chapter
    )


    # --------------------------------------
    # No results
    # --------------------------------------

    if not results:

        return {

            "topic": topic,

            "subject": subject,

            "chapter": chapter,

            "answer":
                "No relevant study material was found for the selected subject/chapter.",

            "sources": []
        }


    # --------------------------------------
    # Build context
    # --------------------------------------

    context = rag.build_context(
        results
    )


    # --------------------------------------
    # Generate progression
    # --------------------------------------

    answer = learning.learning_progression(
        topic,
        context
    )


    return {

        "topic":
            topic,

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
# 15. LEARNING HISTORY
# ==========================================

@app.get("/history")
def get_history():

    return {
        "history":
            learning.get_history()
    }


# ==========================================
# 16. SUBJECTS
# ==========================================

@app.get("/subjects")
def get_subjects():

    return {
        "subjects":
            rag.get_subjects()
    }


# ==========================================
# 17. CHAPTERS
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