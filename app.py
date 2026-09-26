import os
import shutil
import time
import re

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Form
)

from fastapi.responses import (
    FileResponse,
    StreamingResponse
)

from pydantic import BaseModel

from document_processor import process_file
from rag_engine import RAGEngine
from learning_tools import LearningTools

from metadata_manager import (
    load_metadata,
    set_document_metadata,
    get_all_metadata
)


# ==========================================
# 1. FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Subject Guide & Question Bank AI Assistant",
    description="Multi-document RAG academic learning assistant",
    version="3.9"
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
# 4. SUPPORTED FILE TYPES
# ==========================================

SUPPORTED_EXTENSIONS = (
    ".pdf",
    ".docx",
    ".pptx"
)


# ==========================================
# 5. DETECT CONTENT TYPE
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
# 6. QUESTION BANK HELPERS
# ==========================================

def split_question_bank_text(text: str) -> list[str]:
    """
    Split question-bank text into individual questions.

    Supports formats such as:

    1. BFS
    2. DFS
    3) Monkey Banana
    10.Write Tower of Hanoi
    """

    if not text:
        return []

    parts = re.split(
        r"(?=\s*\d+\s*[\.\)])",
        text
    )

    questions = []

    for part in parts:

        part = part.strip()

        if not part:
            continue

        cleaned = re.sub(
            r"^\s*\d+\s*[\.\)]\s*",
            "",
            part
        ).strip()

        if cleaned:
            questions.append(cleaned)

    return questions


def clean_topic_query(query: str) -> str:
    """
    Extract the actual topic from questions such as:

    questions related to BFS
    questions on BFS
    questions about BFS
    give me questions related to BFS
    show me questions about tuples
    """

    topic = query.strip()

    prefixes = [

        "give me questions related to",
        "show me questions related to",
        "list questions related to",
        "what questions are related to",

        "give me question related to",
        "show me question related to",
        "list question related to",
        "what question are related to",

        "give me questions about",
        "show me questions about",
        "list questions about",
        "what questions are about",

        "give me question about",
        "show me question about",
        "list question about",
        "what question are about",

        "give me questions on",
        "show me questions on",
        "list questions on",
        "what questions are on",

        "give me question on",
        "show me question on",
        "list question on",
        "what question are on",

        "questions related to",
        "question related to",

        "questions about",
        "question about",

        "questions on",
        "question on"
    ]

    query_lower = topic.lower()

    for prefix in prefixes:

        if query_lower.startswith(prefix):

            topic = topic[
                len(prefix):
            ].strip()

            break

    # Remove common filler words
    topic = re.sub(
        r"^(the|a|an|about|on)\s+",
        "",
        topic,
        flags=re.IGNORECASE
    ).strip()

    return topic


def get_topic_variations(topic: str) -> list[str]:
    """
    Provide aliases for common technical topics.

    Example:

    BFS
    -> BFS
    -> Breadth First Search
    """

    topic_lower = topic.lower().strip()

    aliases = {

        "bfs": [
            "bfs",
            "breadth first search",
            "breadth-first search"
        ],

        "dfs": [
            "dfs",
            "depth first search",
            "depth-first search"
        ],

        "oop": [
            "oop",
            "object oriented programming",
            "object-oriented programming"
        ],

        "oops": [
            "oops",
            "oop",
            "object oriented programming",
            "object-oriented programming"
        ],

        "dbms": [
            "dbms",
            "database management system"
        ],

        "sql": [
            "sql",
            "structured query language"
        ]
    }

    if topic_lower in aliases:
        return aliases[topic_lower]

    return [topic_lower]


def question_matches_topic(
    question_text: str,
    topic: str
) -> bool:

    question_lower = question_text.lower().strip()

    if not topic:
        return False

    topic_variations = get_topic_variations(
        topic
    )

    # --------------------------------------
    # Exact phrase / alias match
    # --------------------------------------

    for variation in topic_variations:

        if variation in question_lower:
            return True

    # --------------------------------------
    # Multi-word topic
    # --------------------------------------

    topic_words = re.findall(
        r"[a-zA-Z0-9]+",
        topic.lower()
    )

    if len(topic_words) > 1:

        stop_words = {
            "the",
            "a",
            "an",
            "of",
            "and",
            "in",
            "on",
            "to",
            "for",
            "using",
            "with"
        }

        important_words = [
            word
            for word in topic_words
            if word not in stop_words
        ]

        if important_words:

            if all(
                word in question_lower
                for word in important_words
            ):
                return True

    return False


def is_topic_question_query(query: str) -> bool:

    query_lower = query.lower().strip()

    topic_patterns = [

        "questions related to",
        "question related to",

        "questions about",
        "question about",

        "questions on",
        "question on",

        "give me questions related to",
        "show me questions related to",
        "list questions related to",

        "give me questions about",
        "show me questions about",
        "list questions about",

        "give me questions on",
        "show me questions on",
        "list questions on"
    ]

    return any(
        pattern in query_lower
        for pattern in topic_patterns
    )


# ==========================================
# 7. LOAD EXISTING DOCUMENTS
# ==========================================

def load_existing_documents():

    start_time = time.perf_counter()

    print()
    print("==========================================")
    print("LOADING EXISTING DOCUMENTS")
    print("==========================================")

    # --------------------------------------
    # Load persistent metadata
    # --------------------------------------

    document_metadata = load_metadata()

    # --------------------------------------
    # Find supported files
    # --------------------------------------

    supported_files = [
        file
        for file in os.listdir(DATA_FOLDER)
        if file.lower().endswith(
            SUPPORTED_EXTENSIONS
        )
    ]

    print(
        f"Found {len(supported_files)} "
        f"supported document(s)."
    )

    # --------------------------------------
    # Process every document
    # --------------------------------------

    for filename in supported_files:

        file_path = os.path.join(
            DATA_FOLDER,
            filename
        )

        metadata = document_metadata.get(
            filename,
            {
                "subject": "General",
                "chapter": "General",
                "content_type":
                    detect_content_type(
                        filename
                    )
            }
        )

        subject = metadata.get(
            "subject",
            "General"
        )

        chapter = metadata.get(
            "chapter",
            "General"
        )

        content_type = metadata.get(
            "content_type",
            detect_content_type(
                filename
            )
        )

        # Normalize old metadata
        if content_type == "QuestionBank":
            content_type = "Question Bank"

        try:

            document_start = time.perf_counter()

            # ----------------------------------
            # Process document
            # ----------------------------------

            documents = process_file(
                file_path=file_path,
                subject=subject,
                chapter=chapter
            )

            # ----------------------------------
            # Add content type
            # ----------------------------------

            for document in documents:

                document["metadata"][
                    "content_type"
                ] = content_type

            # ----------------------------------
            # Add without rebuilding each time
            # ----------------------------------

            rag.add_documents(
                documents,
                rebuild=False
            )

            # ----------------------------------
            # Save metadata
            # ----------------------------------

            set_document_metadata(
                filename=filename,
                subject=subject,
                chapter=chapter,
                content_type=content_type
            )

            document_time = (
                time.perf_counter()
                - document_start
            )

            print(
                f"Loaded: {filename} "
                f"| Subject: {subject} "
                f"| Chapter: {chapter} "
                f"| Type: {content_type} "
                f"| Chunks: {len(documents)} "
                f"| Time: {document_time:.2f}s"
            )

        except Exception as e:

            print(
                f"Could not load "
                f"{filename}: {e}"
            )

    # --------------------------------------
    # BUILD FAISS ONLY ONCE
    # --------------------------------------

    print()
    print(
        "Building FAISS index once..."
    )

    index_start = time.perf_counter()

    if rag.documents:
        rag._rebuild_index()

    index_time = (
        time.perf_counter()
        - index_start
    )

    total_time = (
        time.perf_counter()
        - start_time
    )

    print(
        f"FAISS build time: "
        f"{index_time:.2f}s"
    )

    print(
        f"Total startup document loading time: "
        f"{total_time:.2f}s"
    )

    print(
        f"Total chunks: "
        f"{len(rag.documents)}"
    )

    print(
        f"Total vectors: "
        f"{rag.get_vector_count()}"
    )

    print(
        "=========================================="
    )


# Load documents at startup
load_existing_documents()


# ==========================================
# 8. REQUEST MODELS
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
# 9. HOME PAGE
# ==========================================

@app.get("/")
def home():

    return FileResponse(
        "static/index.html"
    )


# ==========================================
# 10. HEALTH CHECK
# ==========================================

@app.get("/health")
def health():

    return {

        "status":
            "running",

        "documents":
            rag.get_document_count(),

        "vectors":
            rag.get_vector_count(),

        "subjects":
            rag.get_subjects(),

        "model":
            learning.model,

        "supported_formats": [
            "PDF",
            "DOCX",
            "PPTX"
        ]
    }


# ==========================================
# 11. UPLOAD DOCUMENT
# ==========================================

@app.post("/upload")
async def upload_document(

    file: UploadFile = File(...),

    subject: str = Form(
        "General"
    ),

    chapter: str = Form(
        "General"
    )
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No filename provided."
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

        # ----------------------------------
        # Save file
        # ----------------------------------

        with open(
            file_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        # ----------------------------------
        # Process file
        # ----------------------------------

        documents = process_file(
            file_path=file_path,
            subject=subject,
            chapter=chapter
        )

        # ----------------------------------
        # Save metadata
        # ----------------------------------

        set_document_metadata(
            filename=filename,
            subject=subject,
            chapter=chapter,
            content_type=content_type
        )

        # ----------------------------------
        # Add content type
        # ----------------------------------

        for document in documents:

            document["metadata"][
                "content_type"
            ] = content_type

        # ----------------------------------
        # Add to RAG
        # ----------------------------------

        rag.add_documents(
            documents,
            rebuild=True
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
# 12. ASK QUESTION
# ==========================================

@app.post("/ask")
def ask_question(
    data: Question
):

    request_start = time.perf_counter()

    query = data.question.strip()

    if not query:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    # --------------------------------------
    # Clean filters
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

    query_lower = query.lower()

    # ======================================
    # LEARNING PROGRESSION QUERY
    # ======================================

    learning_query = any(
        phrase in query_lower
        for phrase in [

            "teach me",
            "teach ",
            "learn ",
            "study ",

            "prepare me for",
            "help me prepare",

            "learning progression",

            "theory example practice assessment"
        ]
    )

    # ======================================
    # LIST QUESTION BANK QUERY
    # ======================================

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

            "programs are included in the question bank",

            "show all questions",

            "show the questions"
        ]
    )

    # ======================================
    # TOPIC-WISE QUESTION QUERY
    # ======================================

    topic_question_query = is_topic_question_query(
        query
    )

    # ======================================
    # LIST ALL QUESTION BANK QUESTIONS
    # ======================================

    if list_question_bank_query:

        results = (
            rag.get_all_question_bank_documents(
                subject=subject,
                chapter=chapter
            )
        )

        if not results:

            return {

                "question":
                    query,

                "subject":
                    subject,

                "chapter":
                    chapter,

                "answer":
                    (
                        "No question bank "
                        "material was found "
                        "for the selected "
                        "subject/chapter."
                    ),

                "sources":
                    []
            }

        answer_lines = [
            "Questions in the Question Bank:"
        ]

        question_number = 1

        seen_questions = set()

        source_questions = []

        for result in results:

            text = result.get(
                "text",
                ""
            ).strip()

            if not text:
                continue

            parts = split_question_bank_text(
                text
            )

            for part in parts:

                normalized = part.strip()

                if not normalized:
                    continue

                key = normalized.lower()

                if key in seen_questions:
                    continue

                seen_questions.add(key)

                answer_lines.append(
                    f"{question_number}. "
                    f"{normalized}"
                )

                # Preserve the original result
                # so filename/metadata remains intact.
                source = dict(result)

                source["text"] = normalized

                source_questions.append(
                    source
                )

                question_number += 1

        answer = "\n".join(
            answer_lines
        )

        total_time = (
            time.perf_counter()
            - request_start
        )

        print(
            f"Question Bank listing time: "
            f"{total_time:.2f}s"
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
                source_questions
        }

    # ======================================
    # TOPIC-WISE QUESTION BANK
    # ======================================

    if topic_question_query:

        # ----------------------------------
        # Extract actual topic
        # ----------------------------------

        topic = clean_topic_query(
            query
        )

        print(
            f"Topic question request: "
            f"{topic}"
        )

        # ----------------------------------
        # Dedicated question-bank retrieval
        # ----------------------------------

        results = (
            rag.get_topic_question_bank_documents(
                topic=topic,
                subject=subject,
                chapter=chapter,
                limit=10
            )
        )

        matched_questions = []

        matched_sources = []

        seen_questions = set()

        # ----------------------------------
        # Check every individual question
        # ----------------------------------

        for result in results:

            text = result.get(
                "text",
                ""
            ).strip()

            if not text:
                continue

            parts = split_question_bank_text(
                text
            )

            for part in parts:

                cleaned_question = part.strip()

                if not cleaned_question:
                    continue

                # ----------------------------------
                # ACTUAL TOPIC FILTER
                # ----------------------------------

                if not question_matches_topic(
                    cleaned_question,
                    topic
                ):
                    continue

                key = cleaned_question.lower()

                if key in seen_questions:
                    continue

                seen_questions.add(key)

                matched_questions.append(
                    cleaned_question
                )

                # ----------------------------------
                # IMPORTANT:
                # Preserve the original RAG result.
                # This keeps filename, metadata,
                # subject, chapter, distance, etc.
                # ----------------------------------

                source = dict(result)

                source["text"] = cleaned_question

                matched_sources.append(
                    source
                )

        # ----------------------------------
        # Matching questions found
        # ----------------------------------

        if matched_questions:

            answer_lines = [
                f"Questions related to {topic}:"
            ]

            for i, question_text in enumerate(
                matched_questions,
                start=1
            ):

                answer_lines.append(
                    f"{i}. {question_text}"
                )

            answer = "\n".join(
                answer_lines
            )

            total_time = (
                time.perf_counter()
                - request_start
            )

            print(
                f"Topic question search time: "
                f"{total_time:.2f}s"
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
                    matched_sources
            }

        # ----------------------------------
        # No matching questions
        # ----------------------------------

        total_time = (
            time.perf_counter()
            - request_start
        )

        print(
            f"No exact topic questions found "
            f"in {total_time:.2f}s"
        )

        return {

            "question":
                query,

            "subject":
                subject,

            "chapter":
                chapter,

            "answer":
                (
                    f"I couldn't find a question "
                    f"specifically related to "
                    f"'{topic}' in the uploaded "
                    f"question bank."
                ),

            "sources":
                []
        }

    # ======================================
    # LEARNING PROGRESSION
    # ======================================

    if learning_query:

        # ----------------------------------
        # Extract topic
        # ----------------------------------

        topic = query

        prefixes = [
            "teach me",
            "teach",
            "learn",
            "study",
            "prepare me for",
            "help me prepare"
        ]

        for prefix in prefixes:

            if query_lower.startswith(prefix):

                topic = query[
                    len(prefix):
                ].strip()

                break

        # Remove common filler words

        topic = re.sub(
            r"^(about|on|the)\s+",
            "",
            topic,
            flags=re.IGNORECASE
        ).strip()

        if not topic:

            return {

                "question":
                    query,

                "subject":
                    subject,

                "chapter":
                    chapter,

                "answer":
                    "Please specify a topic to learn.",

                "sources":
                    []
            }

        # ----------------------------------
        # Retrieve study material
        # ----------------------------------

        results = rag.retrieve(
            topic,
            k=5,
            subject=subject,
            chapter=chapter
        )

        # ----------------------------------
        # Retrieve question-bank questions
        # ----------------------------------

        question_results = (
            rag.get_topic_question_bank_documents(
                topic=topic,
                subject=subject,
                chapter=chapter,
                limit=10
            )
        )

        practice_questions = []

        seen_practice = set()

        for result in question_results:

            question_text = result.get(
                "text",
                ""
            ).strip()

            if not question_text:
                continue

            parts = split_question_bank_text(
                question_text
            )

            for part in parts:

                cleaned = part.strip()

                if not cleaned:
                    continue

                if not question_matches_topic(
                    cleaned,
                    topic
                ):
                    continue

                key = cleaned.lower()

                if key in seen_practice:
                    continue

                seen_practice.add(key)

                practice_questions.append(
                    cleaned
                )

        # ----------------------------------
        # No study material
        # ----------------------------------

        if not results:

            return {

                "question":
                    query,

                "subject":
                    subject,

                "chapter":
                    chapter,

                "answer":
                    (
                        "No relevant study "
                        "material was found "
                        "for this topic."
                    ),

                "sources":
                    question_results
            }

        # ----------------------------------
        # Build study context
        # ----------------------------------

        context = rag.build_context(
            results
        )

        # ----------------------------------
        # Generate progression
        # ----------------------------------

        answer = learning.learning_progression(
            topic,
            context,
            practice_questions=practice_questions
        )

        total_time = (
            time.perf_counter()
            - request_start
        )

        print(
            f"Learning progression time: "
            f"{total_time:.2f}s"
        )

        # ----------------------------------
        # Combine sources
        # ----------------------------------

        combined_sources = (
            results + question_results
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
                combined_sources
        }

    # ======================================
    # NORMAL RAG QUESTION
    # ======================================

    results = rag.retrieve(
        query,
        k=5,
        subject=subject,
        chapter=chapter
    )

    # ======================================
    # NO RESULTS
    # ======================================

    if not results:

        return {

            "question":
                query,

            "subject":
                subject,

            "chapter":
                chapter,

            "answer":
                (
                    "No relevant study "
                    "material was found "
                    "for the selected "
                    "subject/chapter."
                ),

            "sources":
                []
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

    total_time = (
        time.perf_counter()
        - request_start
    )

    print(
        f"Total /ask time: "
        f"{total_time:.2f}s"
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
# 13. TOPIC EXPLANATION
# ==========================================

@app.post("/explain")
def explain_topic(
    data: TopicRequest
):

    topic = data.topic.strip()

    if not topic:

        raise HTTPException(
            status_code=400,
            detail="Topic cannot be empty."
        )

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

    results = rag.retrieve(
        topic,
        k=5,
        subject=subject,
        chapter=chapter
    )

    if not results:

        return {

            "topic":
                topic,

            "subject":
                subject,

            "chapter":
                chapter,

            "answer":
                (
                    "No relevant study "
                    "material was found "
                    "for the selected "
                    "subject/chapter."
                ),

            "sources":
                []
        }

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
# 14. CONTENT SYNTHESIS
# ==========================================

@app.post("/synthesize")
def synthesize_content(
    data: TopicRequest
):

    topic = data.topic.strip()

    if not topic:

        raise HTTPException(
            status_code=400,
            detail="Topic cannot be empty."
        )

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

    results = rag.retrieve(
        topic,
        k=5,
        subject=subject,
        chapter=chapter
    )

    if not results:

        return {

            "topic":
                topic,

            "subject":
                subject,

            "chapter":
                chapter,

            "answer":
                (
                    "No relevant study "
                    "material was found "
                    "for the selected "
                    "subject/chapter."
                ),

            "sources":
                []
        }

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
# 15. LEARNING PROGRESSION
# ==========================================

@app.post("/progression")
def progression(
    data: TopicRequest
):

    topic = data.topic.strip()

    if not topic:

        raise HTTPException(
            status_code=400,
            detail="Topic cannot be empty."
        )

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
    # Retrieve study material
    # --------------------------------------

    results = rag.retrieve(
        topic,
        k=5,
        subject=subject,
        chapter=chapter
    )

    # --------------------------------------
    # Retrieve question-bank questions
    # --------------------------------------

    question_results = (
        rag.get_topic_question_bank_documents(
            topic=topic,
            subject=subject,
            chapter=chapter,
            limit=10
        )
    )

    practice_questions = []

    seen_practice = set()

    for result in question_results:

        question_text = result.get(
            "text",
            ""
        ).strip()

        if not question_text:
            continue

        parts = split_question_bank_text(
            question_text
        )

        for part in parts:

            cleaned = part.strip()

            if not cleaned:
                continue

            if not question_matches_topic(
                cleaned,
                topic
            ):
                continue

            key = cleaned.lower()

            if key in seen_practice:
                continue

            seen_practice.add(key)

            practice_questions.append(
                cleaned
            )

    # --------------------------------------
    # No study material
    # --------------------------------------

    if not results:

        return {

            "topic":
                topic,

            "subject":
                subject,

            "chapter":
                chapter,

            "answer":
                (
                    "No relevant study "
                    "material was found "
                    "for the selected "
                    "subject/chapter."
                ),

            "sources":
                question_results
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
        context,
        practice_questions=practice_questions
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
            results + question_results
    }


# ==========================================
# 16. LEARNING HISTORY
# ==========================================

@app.get("/history")
def get_history():

    return {
        "history":
            learning.get_history()
    }


# ==========================================
# 17. SUBJECTS
# ==========================================

@app.get("/subjects")
def get_subjects():

    return {
        "subjects":
            rag.get_subjects()
    }


# ==========================================
# 18. CHAPTERS
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


# ==========================================
# 19. DOCUMENT BROWSER
# ==========================================

@app.get("/documents")
def get_documents():

    metadata = get_all_metadata()

    documents = []

    for filename, info in metadata.items():

        file_path = os.path.join(
            DATA_FOLDER,
            filename
        )

        exists = os.path.exists(
            file_path
        )

        content_type = info.get(
            "content_type",
            "Notes"
        )

        if content_type == "QuestionBank":
            content_type = "Question Bank"

        documents.append({

            "filename":
                filename,

            "subject":
                info.get(
                    "subject",
                    "General"
                ),

            "chapter":
                info.get(
                    "chapter",
                    "General"
                ),

            "content_type":
                content_type,

            "format":
                os.path.splitext(
                    filename
                )[1].replace(
                    ".",
                    ""
                ).upper(),

            "exists":
                exists,

            "size_bytes":
                (
                    os.path.getsize(
                        file_path
                    )
                    if exists
                    else 0
                )
        })

    return {
        "documents":
            documents
    }


# ==========================================
# 20. OPEN DOCUMENT
# ==========================================

@app.get(
    "/document/{filename:path}"
)
def open_document(
    filename: str
):

    file_path = os.path.join(
        DATA_FOLDER,
        filename
    )

    # --------------------------------------
    # Security check
    # --------------------------------------

    data_folder_abs = os.path.abspath(
        DATA_FOLDER
    )

    file_path_abs = os.path.abspath(
        file_path
    )

    try:

        common_path = os.path.commonpath([
            data_folder_abs,
            file_path_abs
        ])

    except ValueError:

        raise HTTPException(
            status_code=400,
            detail="Invalid file path."
        )

    if common_path != data_folder_abs:

        raise HTTPException(
            status_code=400,
            detail="Invalid file path."
        )

    # --------------------------------------
    # File existence
    # --------------------------------------

    if not os.path.isfile(
        file_path_abs
    ):

        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    # --------------------------------------
    # Only PDF preview
    # --------------------------------------

    extension = os.path.splitext(
        file_path_abs
    )[1].lower()

    if extension != ".pdf":

        raise HTTPException(
            status_code=400,
            detail=(
                "Only PDF documents can "
                "be opened directly in "
                "the browser."
            )
        )

    # --------------------------------------
    # Stream PDF
    # --------------------------------------

    def file_iterator():

        with open(
            file_path_abs,
            "rb"
        ) as file:

            while True:

                chunk = file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                yield chunk

    return StreamingResponse(

        file_iterator(),

        media_type="application/pdf",

        headers={
            "Content-Disposition":
                (
                    'inline; filename="'
                    + os.path.basename(
                        file_path_abs
                    )
                    + '"'
                )
        }
    )