import json
import os
import re
import shutil
import sys
import time

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from document_processor import process_file
from rag_engine import RAGEngine
from learning_tools import LearningTools
from metadata_manager import (
    load_metadata,
    set_document_metadata,
    get_all_metadata,
)


app = FastAPI(
    title="Subject Guide & Question Bank AI Assistant",
    description="Multi-document RAG academic learning assistant",
    version="3.11",
)


rag = RAGEngine()
learning = LearningTools(model="llama3.2:3b")


BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

DATA_FOLDER = os.getenv(
    "DATA_FOLDER",
    os.path.join(PROJECT_ROOT, "data"),
)
STATIC_FOLDER = os.getenv(
    "STATIC_FOLDER",
    os.path.join(PROJECT_ROOT, "static"),
)
CACHE_FOLDER = os.getenv(
    "CACHE_FOLDER",
    os.path.join(DATA_FOLDER, ".rag_cache"),
)
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(CACHE_FOLDER, exist_ok=True)


SUPPORTED_EXTENSIONS = (
    ".pdf",
    ".docx",
    ".pptx",
)


def detect_content_type(filename: str) -> str:
    name = filename.lower()

    if any(
        k in name
        for k in [
            "question",
            "question bank",
            "questionbank",
            "previous year",
            "previousyear",
            "pyq",
            "paper",
            "exam",
            "questionpaper",
            "question paper",
        ]
    ):
        return "Question Bank"

    if any(
        k in name
        for k in [
            "lab",
            "laboratory",
            "practical",
        ]
    ):
        return "Lab Manual"

    if any(
        k in name
        for k in [
            "textbook",
            "book",
        ]
    ):
        return "Textbook"

    return "Notes"


def is_question_bank_result(result: dict) -> bool:
    """Return True when a retrieved result belongs to a question bank."""

    content_type = str(
        result.get("content_type")
        or result.get(
            "metadata",
            {},
        ).get(
            "content_type",
            "",
        )
    ).strip().lower()

    filename = str(
        result.get(
            "filename",
            "",
        )
    ).lower()

    return (
        content_type
        in {
            "question bank",
            "questionbank",
        }
        or "question bank" in filename
        or "questionbank" in filename
        or "questionpaper" in filename
        or "previous year" in filename
        or "pyq" in filename
    )


def get_study_results(
    results: list[dict],
) -> list[dict]:
    """Keep study material and exclude question-bank chunks."""

    return [
        result
        for result in results
        if not is_question_bank_result(result)
    ]


def merge_document_results(
    *result_sets: list[dict],
) -> list[dict]:
    """Merge retrieved results without duplicating the same chunk."""

    merged = []
    seen = set()

    for result_set in result_sets:
        for result in result_set:

            key = (
                result.get("filename"),
                result.get("index"),
                result.get("text", ""),
            )

            if key in seen:
                continue

            seen.add(key)
            merged.append(result)

    return merged


def get_example_source_context(
    study_results: list[dict],
    subject: str | None = None,
    chapter: str | None = None,
    max_source_files: int = 2,
) -> str:
    """
    Collect all study chunks from the most relevant source files.

    This gives the example extractor enough neighboring material
    to reconstruct a complete code example when a PDF splits
    it across multiple vector chunks.
    """

    if not study_results:
        return ""

    source_files = []
    seen_files = set()

    for result in study_results:

        filename = result.get(
            "filename"
        )

        if (
            not filename
            or filename in seen_files
        ):
            continue

        seen_files.add(filename)
        source_files.append(filename)

        if (
            len(source_files)
            >= max_source_files
        ):
            break

    selected = []

    for document in rag.documents:

        metadata = document.get(
            "metadata",
            {},
        )

        filename = (
            document.get("filename")
            or metadata.get("filename")
        )

        if filename not in source_files:
            continue

        content_type = str(
            document.get("content_type")
            or metadata.get(
                "content_type",
                "",
            )
        ).strip().lower()

        if content_type in {
            "question bank",
            "questionbank",
        }:
            continue

        document_subject = (
            document.get("subject")
            or metadata.get("subject")
        )

        document_chapter = (
            document.get("chapter")
            or metadata.get("chapter")
        )

        if (
            subject
            and document_subject != subject
        ):
            continue

        if (
            chapter
            and document_chapter != chapter
        ):
            continue

        selected.append(document)

    selected.sort(
        key=lambda item: item.get(
            "index",
            10**9,
        )
    )

    if not selected:
        return ""

    return rag.build_context(
        selected
    )


def split_question_bank_text(
    text: str,
) -> list[str]:

    parts = re.split(
        r"(?=\b\d{1,3}\s*[\.\)])",
        text,
    )

    questions = []

    for part in parts:

        part = part.strip()

        if not part:
            continue

        cleaned = re.sub(
            r"^\s*\d{1,3}\s*[\.\)]\s*",
            "",
            part,
        ).strip()

        if cleaned:
            questions.append(
                cleaned
            )

    return questions


def clean_topic_query(
    query: str,
) -> str:

    topic = query.strip()

    prefixes = [
        "give me questions related to",
        "show me questions related to",
        "list questions related to",
        "what questions are related to",
        "give me questions about",
        "show me questions about",
        "list questions about",
        "what questions are about",
        "give me questions on",
        "show me questions on",
        "list questions on",
        "what questions are on",
        "give me question related to",
        "questions related to",
        "question related to",
        "questions about",
        "question about",
        "questions on",
        "question on",
    ]

    for prefix in prefixes:

        if topic.lower().startswith(
            prefix
        ):

            topic = topic[
                len(prefix):
            ].strip()

            break

    return topic.strip(
        " :?-"
    )


def question_matches_topic(
    question_text: str,
    topic: str,
) -> bool:

    q = question_text.lower().strip()
    t = topic.lower().strip()

    if not t:
        return False

    aliases = {
        "bfs": [
            "breadth first search",
            "bfs",
        ],
        "breadth first search": [
            "breadth first search",
            "bfs",
        ],
        "dfs": [
            "depth first search",
            "dfs",
        ],
        "depth first search": [
            "depth first search",
            "dfs",
        ],
    }

    candidates = aliases.get(
        t,
        [t],
    )

    for candidate in candidates:

        if candidate in q:
            return True

    words = re.findall(
        r"[a-zA-Z0-9]+",
        t,
    )

    if len(words) > 1:

        return all(
            word in q
            for word in words
            if len(word) > 1
        )

    return False


def extract_matching_questions(
    results: list[dict],
    topic: str,
    limit: int = 10,
):

    matched_questions = []
    matched_sources = []

    seen = set()

    for result in results:

        text = result.get(
            "text",
            "",
        ).strip()

        if not text:
            continue

        for question_text in split_question_bank_text(
            text
        ):

            if not question_matches_topic(
                question_text,
                topic,
            ):
                continue

            key = re.sub(
                r"\s+",
                " ",
                question_text.lower(),
            ).strip()

            if key in seen:
                continue

            seen.add(key)

            matched_questions.append(
                question_text
            )

            matched_sources.append(
                {
                    "filename": result.get(
                        "filename"
                    ),
                    "subject": result.get(
                        "subject"
                    ),
                    "chapter": result.get(
                        "chapter"
                    ),
                    "content_type": result.get(
                        "content_type"
                    ),
                    "distance": result.get(
                        "distance"
                    ),
                    "text": question_text,
                }
            )

            if (
                len(matched_questions)
                >= limit
            ):
                return (
                    matched_questions,
                    matched_sources,
                )

    return (
        matched_questions,
        matched_sources,
    )


def get_documents_manifest():
    """Return dict of filename -> {mtime, size} to track document modifications."""
    manifest = {}
    if not os.path.exists(DATA_FOLDER):
        return manifest
    for f in os.listdir(DATA_FOLDER):
        if f.lower().endswith(SUPPORTED_EXTENSIONS):
            fp = os.path.join(DATA_FOLDER, f)
            try:
                stat = os.stat(fp)
                manifest[f] = {
                    "mtime": stat.st_mtime,
                    "size": stat.st_size,
                }
            except OSError:
                pass
    return manifest


def load_existing_documents():

    start_time = time.perf_counter()

    print(
        "\n=========================================="
    )
    print(
        "LOADING EXISTING DOCUMENTS"
    )
    print(
        "=========================================="
    )

    current_manifest = get_documents_manifest()
    manifest_file = os.path.join(
        CACHE_FOLDER,
        "manifest.json"
    )
    cache_valid = False

    if os.path.exists(manifest_file):
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                cached_manifest = json.load(f)
            if cached_manifest == current_manifest:
                cache_valid = True
        except Exception:
            cache_valid = False

    if cache_valid and rag.load_cache(CACHE_FOLDER):
        print(
            f"FAISS index & documents loaded from cache in "
            f"{time.perf_counter() - start_time:.2f}s!"
        )
        print(
            f"Total chunks: {len(rag.documents)}"
        )
        print(
            f"Total vectors: {rag.get_vector_count()}"
        )
        print(
            "=========================================="
        )
        return

    print("Cache missing or documents updated. Processing files...")

    document_metadata = load_metadata()

    supported_files = list(current_manifest.keys())

    print(
        f"Found {len(supported_files)} supported document(s)."
    )

    for filename in supported_files:

        file_path = os.path.join(
            DATA_FOLDER,
            filename,
        )

        metadata = document_metadata.get(
            filename,
            {
                "subject": "General",
                "chapter": "General",
                "content_type": detect_content_type(
                    filename
                ),
            },
        )

        subject = metadata.get(
            "subject",
            "General",
        )

        chapter = metadata.get(
            "chapter",
            "General",
        )

        content_type = metadata.get(
            "content_type",
            detect_content_type(
                filename
            ),
        )

        if content_type == "QuestionBank":
            content_type = "Question Bank"

        try:

            document_start = (
                time.perf_counter()
            )

            documents = process_file(
                file_path=file_path,
                subject=subject,
                chapter=chapter,
            )

            for document in documents:

                document["metadata"][
                    "content_type"
                ] = content_type

            rag.add_documents(
                documents,
                rebuild=False,
            )

            set_document_metadata(
                filename=filename,
                subject=subject,
                chapter=chapter,
                content_type=content_type,
            )

            document_time = (
                time.perf_counter()
                - document_start
            )

            print(
                f"Loaded: {filename} | "
                f"Subject: {subject} | "
                f"Chapter: {chapter} | "
                f"Type: {content_type} | "
                f"Chunks: {len(documents)} | "
                f"Time: {document_time:.2f}s"
            )

        except Exception as e:

            print(
                f"Could not load {filename}: {e}"
            )

    print(
        "Building FAISS index once..."
    )

    index_start = time.perf_counter()

    if rag.documents:
        rag._rebuild_index()
        rag.save_cache(CACHE_FOLDER)
        try:
            with open(manifest_file, "w", encoding="utf-8") as f:
                json.dump(current_manifest, f)
        except Exception as e:
            print(f"Warning: Failed to save cache manifest: {e}")

    print(
        f"FAISS build time: "
        f"{time.perf_counter() - index_start:.2f}s"
    )

    print(
        f"Total startup document loading time: "
        f"{time.perf_counter() - start_time:.2f}s"
    )

    print(
        f"Total chunks: {len(rag.documents)}"
    )

    print(
        f"Total vectors: {rag.get_vector_count()}"
    )

    print(
        "=========================================="
    )


load_existing_documents()


class Question(BaseModel):
    question: str
    subject: str | None = None
    chapter: str | None = None


class TopicRequest(BaseModel):
    topic: str
    subject: str | None = None
    chapter: str | None = None


class StudyPlanRequest(BaseModel):
    subject: str
    chapter: str = "General"
    topics: list[str] | None = None
    days: int = 5


class ExamPrepRequest(BaseModel):
    topic: str
    subject: str | None = None
    chapter: str | None = None


class QuizRequest(BaseModel):
    topic: str
    subject: str | None = None
    chapter: str | None = None


@app.get("/")
def home():

    return FileResponse(
        os.path.join(
            STATIC_FOLDER,
            "index.html",
        )
    )


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
            "PPTX",
        ],
    }


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    subject: str = Form("General"),
    chapter: str = Form("General"),
    content_type: str | None = Form(None),
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    filename = file.filename

    if not filename.lower().endswith(
        SUPPORTED_EXTENSIONS
    ):

        raise HTTPException(
            status_code=400,
            detail="Supported formats: PDF, DOCX, PPTX.",
        )

    file_path = os.path.join(
        DATA_FOLDER,
        filename,
    )

    final_content_type = (
        content_type.strip()
        if content_type and content_type.strip()
        else detect_content_type(filename)
    )

    try:

        with open(
            file_path,
            "wb",
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer,
            )

        documents = process_file(
            file_path=file_path,
            subject=subject,
            chapter=chapter,
        )

        for document in documents:

            document["metadata"][
                "content_type"
            ] = final_content_type

        set_document_metadata(
            filename=filename,
            subject=subject,
            chapter=chapter,
            content_type=final_content_type,
        )

        rag.add_documents(
            documents,
            rebuild=True,
        )

        rag.save_cache(CACHE_FOLDER)
        try:
            with open(
                os.path.join(CACHE_FOLDER, "manifest.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(get_documents_manifest(), f)
        except Exception:
            pass

        return {
            "message": (
                "Document uploaded successfully"
            ),
            "filename": filename,
            "subject": subject,
            "chapter": chapter,
            "content_type": final_content_type,
            "chunks_added": len(documents),
            "total_documents": (
                rag.get_document_count()
            ),
            "total_vectors": (
                rag.get_vector_count()
            ),
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


@app.post("/upload-batch")
async def upload_documents(
    files: list[UploadFile] = File(default=[]),
    subject: str = Form("General"),
    chapter: str = Form("General"),
    content_type: str | None = Form(None),
):
    valid_files = [f for f in files if f.filename]
    if not valid_files:
        raise HTTPException(
            status_code=400,
            detail="No files uploaded.",
        )

    all_new_documents = []
    processed_files = []

    try:
        for file in files:
            if not file.filename:
                continue

            filename = file.filename
            if not filename.lower().endswith(SUPPORTED_EXTENSIONS):
                continue

            file_path = os.path.join(
                DATA_FOLDER,
                filename,
            )

            final_content_type = (
                content_type.strip()
                if content_type and content_type.strip()
                else detect_content_type(filename)
            )

            with open(
                file_path,
                "wb",
            ) as buffer:
                shutil.copyfileobj(
                    file.file,
                    buffer,
                )

            documents = process_file(
                file_path=file_path,
                subject=subject,
                chapter=chapter,
            )

            for document in documents:
                document["metadata"][
                    "content_type"
                ] = final_content_type

            set_document_metadata(
                filename=filename,
                subject=subject,
                chapter=chapter,
                content_type=final_content_type,
            )

            all_new_documents.extend(documents)
            processed_files.append({
                "filename": filename,
                "chunks_added": len(documents),
                "content_type": final_content_type,
            })

        if not processed_files:
            raise HTTPException(
                status_code=400,
                detail="No valid PDF, DOCX, or PPTX files were found in the upload.",
            )

        if all_new_documents:
            rag.add_documents(
                all_new_documents,
                rebuild=True,
            )
            rag.save_cache(CACHE_FOLDER)
            try:
                with open(
                    os.path.join(CACHE_FOLDER, "manifest.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(get_documents_manifest(), f)
            except Exception:
                pass

        return {
            "message": f"Successfully uploaded and indexed {len(processed_files)} document(s).",
            "files": processed_files,
            "total_chunks_added": len(all_new_documents),
            "total_documents": rag.get_document_count(),
            "total_vectors": rag.get_vector_count(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


@app.post("/ask")
def ask_question(
    data: Question,
):

    request_start = (
        time.perf_counter()
    )

    query = data.question.strip()

    if not query:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
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

    query_lower = query.lower()

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
            "theory example practice assessment",
        ]
    )

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
            "show the questions",
        ]
    )

    topic_question_query = any(
        phrase in query_lower
        for phrase in [
            "questions related to",
            "question related to",
            "questions about",
            "question about",
            "questions on",
            "question on",
        ]
    )

    # ==========================================================
    # LIST ALL QUESTION-BANK QUESTIONS
    # ==========================================================

    if list_question_bank_query:

        results = (
            rag.get_all_question_bank_documents(
                subject=subject,
                chapter=chapter,
            )
        )

        if not results:

            return {
                "question": query,
                "subject": subject,
                "chapter": chapter,
                "answer": (
                    "No question bank material was found "
                    "for the selected subject/chapter."
                ),
                "sources": [],
            }

        answer_lines = [
            "Questions in the Question Bank:"
        ]

        sources = []

        seen = set()

        question_number = 1

        for result in results:

            for question_text in split_question_bank_text(
                result.get(
                    "text",
                    "",
                )
            ):

                key = re.sub(
                    r"\s+",
                    " ",
                    question_text.lower(),
                ).strip()

                if key in seen:
                    continue

                seen.add(key)

                answer_lines.append(
                    f"{question_number}. {question_text}"
                )

                sources.append(
                    {
                        "filename": result.get(
                            "filename"
                        ),
                        "subject": result.get(
                            "subject"
                        ),
                        "chapter": result.get(
                            "chapter"
                        ),
                        "content_type": result.get(
                            "content_type"
                        ),
                        "distance": result.get(
                            "distance"
                        ),
                        "text": question_text,
                    }
                )

                question_number += 1

        return {
            "question": query,
            "subject": subject,
            "chapter": chapter,
            "answer": "\n".join(
                answer_lines
            ),
            "sources": sources,
        }

    # ==========================================================
    # TOPIC-WISE QUESTION-BANK QUESTIONS
    # ==========================================================

    if topic_question_query:

        topic = clean_topic_query(
            query
        )

        if not topic:

            return {
                "question": query,
                "subject": subject,
                "chapter": chapter,
                "answer": (
                    "Please specify a topic."
                ),
                "sources": [],
            }

        results = (
            rag.get_topic_question_bank_documents(
                topic=topic,
                subject=subject,
                chapter=chapter,
                limit=10,
            )
        )

        (
            matched_questions,
            matched_sources,
        ) = extract_matching_questions(
            results,
            topic,
            limit=10,
        )

        if not matched_questions:

            return {
                "question": query,
                "subject": subject,
                "chapter": chapter,
                "answer": (
                    f"No questions specifically related "
                    f"to '{topic}' were found in the "
                    f"uploaded question bank."
                ),
                "sources": [],
            }

        answer_lines = [
            f"Questions related to {topic}:"
        ]

        for i, question_text in enumerate(
            matched_questions,
            1,
        ):

            answer_lines.append(
                f"{i}. {question_text}"
            )

        return {
            "question": query,
            "subject": subject,
            "chapter": chapter,
            "answer": "\n".join(
                answer_lines
            ),
            "sources": matched_sources,
        }

    # ==========================================================
    # LEARNING PROGRESSION
    # ==========================================================

    if learning_query:

        topic = query

        prefixes = [
            "teach me",
            "teach",
            "learn",
            "study",
            "prepare me for",
            "help me prepare",
        ]

        for prefix in prefixes:

            if query_lower.startswith(
                prefix
            ):

                topic = query[
                    len(prefix):
                ].strip()

                break

        topic = re.sub(
            r"^(about|on|the)\s+",
            "",
            topic,
            flags=re.IGNORECASE,
        ).strip()

        if not topic:

            return {
                "question": query,
                "subject": subject,
                "chapter": chapter,
                "answer": (
                    "Please specify a topic to learn."
                ),
                "sources": [],
            }

        results = rag.retrieve(
            topic,
            k=12,
            subject=subject,
            chapter=chapter,
        )

        example_results = rag.retrieve(
            f"{topic} Python code example implementation",
            k=15,
            subject=subject,
            chapter=chapter,
        )

        combined_results = (
            merge_document_results(
                results,
                example_results,
            )
        )

        study_results = get_study_results(
            combined_results
        )

        question_results = (
            rag.get_topic_question_bank_documents(
                topic=topic,
                subject=subject,
                chapter=chapter,
                limit=10,
            )
        )

        (
            practice_questions,
            _,
        ) = extract_matching_questions(
            question_results,
            topic,
            limit=10,
        )

        if not study_results:

            practice = (
                "\n".join(
                    f"{i}. {question}"
                    for i, question in enumerate(
                        practice_questions,
                        start=1,
                    )
                )
                if practice_questions
                else (
                    "No related practice questions "
                    "were found in the uploaded "
                    "question bank."
                )
            )

            assessment = (
                "\n".join(
                    f"{i}. {question}"
                    for i, question in enumerate(
                        practice_questions[:3],
                        start=1,
                    )
                )
                if practice_questions
                else (
                    "No separate assessment question "
                    "is available in the uploaded "
                    "question bank."
                )
            )

            answer_parts = [
                "THEORY",
                "",
                (
                    "No supporting study material was "
                    "found for this topic in the selected "
                    "subject/chapter."
                ),
                "",
                "EXAMPLE",
                "",
                (
                    "No example was found in the "
                    "uploaded study material."
                ),
                "",
                "PRACTICE",
                "",
                practice,
                "",
                "ASSESSMENT",
                "",
                assessment,
            ]

            return {
                "question": query,
                "subject": subject,
                "chapter": chapter,
                "answer": "\n".join(
                    answer_parts
                ),
                "sources": question_results,
            }

        context = rag.build_context(
            study_results[:12]
        )

        # ======================================================
        # IMPORTANT FIX:
        #
        # Do NOT use only the first 20 retrieved chunks.
        #
        # Instead, collect ALL chunks belonging to the
        # relevant source document(s). This allows the BFS
        # example extractor to recover the complete example
        # even when the code is split across many RAG chunks.
        # ======================================================

        example_context = (
            get_example_source_context(
                study_results,
                subject=subject,
                chapter=chapter,
                max_source_files=2,
            )
        )

        answer = learning.learning_progression(
            topic,
            context,
            practice_questions=practice_questions,
            example_context=example_context,
        )

        print(
            f"Learning progression time: "
            f"{time.perf_counter() - request_start:.2f}s"
        )

        return {
            "topic": topic,
            "subject": subject,
            "chapter": chapter,
            "answer": answer,
            "sources": (
                study_results
                + question_results
            ),
        }

    # ==========================================================
    # NORMAL RAG QUESTION
    # ==========================================================

    results = rag.retrieve(
        query,
        k=12,
        subject=subject,
        chapter=chapter,
    )

    study_results = get_study_results(
        results
    )

    if not study_results:

        return {
            "question": query,
            "subject": subject,
            "chapter": chapter,
            "answer": (
                "No relevant study material was "
                "found for the selected subject/chapter."
            ),
            "sources": [],
        }

    context = rag.build_context(
        study_results[:12]
    )

    example_context = (
        get_example_source_context(
            study_results,
            subject=subject,
            chapter=chapter,
            max_source_files=2,
        )
    )

    answer = learning.solve_question(
        query,
        context,
        example_context=example_context,
    )

    print(
        f"Total /ask time: "
        f"{time.perf_counter() - request_start:.2f}s"
    )

    return {
        "question": query,
        "subject": subject,
        "chapter": chapter,
        "answer": answer,
        "sources": study_results,
    }


@app.post("/explain")
def explain_topic(
    data: TopicRequest,
):

    topic = data.topic.strip()

    if not topic:

        raise HTTPException(
            status_code=400,
            detail="Topic cannot be empty.",
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
        k=8,
        subject=subject,
        chapter=chapter,
    )

    results = get_study_results(
        results
    )

    if not results:

        return {
            "topic": topic,
            "subject": subject,
            "chapter": chapter,
            "answer": (
                "No relevant study material was "
                "found for the selected subject/chapter."
            ),
            "sources": [],
        }

    context = rag.build_context(
        results
    )

    answer = learning.explain_topic(
        topic,
        context,
    )

    return {
        "topic": topic,
        "subject": subject,
        "chapter": chapter,
        "answer": answer,
        "sources": results,
    }


@app.post("/synthesize")
def synthesize_content(
    data: TopicRequest,
):

    topic = data.topic.strip()

    if not topic:

        raise HTTPException(
            status_code=400,
            detail="Topic cannot be empty.",
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
        k=8,
        subject=subject,
        chapter=chapter,
    )

    results = get_study_results(
        results
    )

    if not results:

        return {
            "topic": topic,
            "subject": subject,
            "chapter": chapter,
            "answer": (
                "No relevant study material was "
                "found for the selected subject/chapter."
            ),
            "sources": [],
        }

    context = rag.build_context(
        results
    )

    answer = learning.synthesize_content(
        topic,
        context,
    )

    return {
        "topic": topic,
        "subject": subject,
        "chapter": chapter,
        "answer": answer,
        "sources": results,
    }


@app.post("/progression")
def progression(
    data: TopicRequest,
):

    request_start = (
        time.perf_counter()
    )

    topic = data.topic.strip()

    if not topic:

        raise HTTPException(
            status_code=400,
            detail="Topic cannot be empty.",
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
        k=12,
        subject=subject,
        chapter=chapter,
    )

    example_results = rag.retrieve(
        f"{topic} Python code example implementation",
        k=15,
        subject=subject,
        chapter=chapter,
    )

    combined_results = (
        merge_document_results(
            results,
            example_results,
        )
    )

    study_results = get_study_results(
        combined_results
    )

    question_results = (
        rag.get_topic_question_bank_documents(
            topic=topic,
            subject=subject,
            chapter=chapter,
            limit=10,
        )
    )

    (
        practice_questions,
        _,
    ) = extract_matching_questions(
        question_results,
        topic,
        limit=10,
    )

    if not study_results:

        practice = (
            "\n".join(
                f"{i}. {question}"
                for i, question in enumerate(
                    practice_questions,
                    start=1,
                )
            )
            if practice_questions
            else (
                "No related practice questions "
                "were found in the uploaded question bank."
            )
        )

        assessment = (
            "\n".join(
                f"{i}. {question}"
                for i, question in enumerate(
                    practice_questions[:3],
                    start=1,
                )
            )
            if practice_questions
            else (
                "No separate assessment question "
                "is available in the uploaded "
                "question bank."
            )
        )

        answer = "\n".join(
            [
                "THEORY",
                "",
                (
                    "No supporting study material was "
                    "found for this topic in the selected "
                    "subject/chapter."
                ),
                "",
                "EXAMPLE",
                "",
                (
                    "No example was found in the "
                    "uploaded study material."
                ),
                "",
                "PRACTICE",
                "",
                practice,
                "",
                "ASSESSMENT",
                "",
                assessment,
            ]
        )

        return {
            "topic": topic,
            "subject": subject,
            "chapter": chapter,
            "answer": answer,
            "sources": question_results,
        }

    context = rag.build_context(
        study_results[:12]
    )

    # ==========================================================
    # FIXED:
    #
    # Previously the progression endpoint used:
    #
    # ordered_for_example = sorted(...)
    # example_context = rag.build_context(
    #     ordered_for_example[:20]
    # )
    #
    # That only provided the first 20 retrieved chunks and
    # caused the BFS example to be incomplete.
    #
    # Now we collect ALL chunks belonging to the relevant
    # source document(s).
    # ==========================================================

    example_context = (
        get_example_source_context(
            study_results,
            subject=subject,
            chapter=chapter,
            max_source_files=2,
        )
    )

    answer = learning.learning_progression(
        topic,
        context,
        practice_questions=practice_questions,
        example_context=example_context,
    )

    print(
        f"Learning progression time: "
        f"{time.perf_counter() - request_start:.2f}s"
    )

    return {
        "topic": topic,
        "subject": subject,
        "chapter": chapter,
        "answer": answer,
        "sources": (
            study_results
            + question_results
        ),
    }


@app.post("/exam-prep")
def exam_prep(
    data: ExamPrepRequest,
):
    request_start = time.perf_counter()
    topic = data.topic.strip()

    if not topic:
        raise HTTPException(
            status_code=400,
            detail="Topic cannot be empty.",
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
        k=12,
        subject=subject,
        chapter=chapter,
    )

    study_results = get_study_results(results)

    question_results = rag.get_topic_question_bank_documents(
        topic=topic,
        subject=subject,
        chapter=chapter,
        limit=10,
    )

    practice_questions, _ = extract_matching_questions(
        question_results,
        topic,
        limit=10,
    )

    context = rag.build_context(
        study_results[:12]
    )

    example_context = get_example_source_context(
        study_results,
        subject=subject,
        chapter=chapter,
        max_source_files=2,
    )

    answer = learning.exam_prep_guide(
        topic=topic,
        context=context,
        practice_questions=practice_questions,
        example_context=example_context,
    )

    print(
        f"Exam prep guide time: "
        f"{time.perf_counter() - request_start:.2f}s"
    )

    return {
        "topic": topic,
        "subject": subject,
        "chapter": chapter,
        "answer": answer,
        "sources": study_results + question_results,
    }


@app.post("/study-plan")
def create_study_plan(
    data: StudyPlanRequest,
):
    request_start = time.perf_counter()
    subject = data.subject.strip()
    chapter = data.chapter.strip() if data.chapter else "General"
    days = max(1, min(14, data.days))

    results = rag.retrieve(
        f"{subject} {chapter} concepts topics syllabus",
        k=8,
        subject=subject,
    )
    study_results = get_study_results(results)
    context = rag.build_context(study_results)

    question_results = rag.get_all_question_bank_documents(
        subject=subject,
        chapter=chapter,
    )
    practice_questions = [
        r.get("text", "")
        for r in question_results[:10]
    ]

    topics = data.topics or rag.get_chapters(subject=subject)

    plan = learning.generate_study_plan(
        subject=subject,
        chapter=chapter,
        topics=topics,
        days=days,
        context=context,
        practice_questions=practice_questions,
    )

    print(
        f"Study plan time: "
        f"{time.perf_counter() - request_start:.2f}s"
    )

    return {
        "subject": subject,
        "chapter": chapter,
        "days": days,
        "answer": plan,
        "sources": study_results,
    }


@app.post("/quiz")
def generate_topic_quiz(
    data: QuizRequest,
):
    request_start = time.perf_counter()
    topic = data.topic.strip()

    if not topic:
        raise HTTPException(
            status_code=400,
            detail="Topic cannot be empty.",
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
        k=10,
        subject=subject,
        chapter=chapter,
    )

    study_results = get_study_results(results)

    if not study_results:
        raise HTTPException(
            status_code=404,
            detail=f"No study material found for topic '{topic}' to build a quiz.",
        )

    context = rag.build_context(study_results)

    quiz = learning.generate_quiz(
        topic=topic,
        context=context,
    )

    print(
        f"Quiz generation time: "
        f"{time.perf_counter() - request_start:.2f}s"
    )

    return {
        "topic": topic,
        "subject": subject,
        "chapter": chapter,
        "answer": quiz,
        "sources": study_results,
    }


@app.get("/history")
def get_history():

    return {
        "history": learning.get_history()
    }


@app.get("/subjects")
def get_subjects():

    return {
        "subjects": rag.get_subjects()
    }


@app.get("/chapters")
def get_chapters(
    subject: str | None = None,
):

    return {
        "subject": subject,
        "chapters": rag.get_chapters(
            subject
        ),
    }


@app.get("/documents")
def get_documents():

    metadata = get_all_metadata()

    documents = []

    for filename, info in metadata.items():

        file_path = os.path.join(
            DATA_FOLDER,
            filename,
        )

        exists = os.path.exists(
            file_path
        )

        content_type = info.get(
            "content_type",
            "Notes",
        )

        if content_type == "QuestionBank":
            content_type = "Question Bank"

        documents.append(
            {
                "filename": filename,
                "subject": info.get(
                    "subject",
                    "General",
                ),
                "chapter": info.get(
                    "chapter",
                    "General",
                ),
                "content_type": content_type,
                "format": (
                    os.path.splitext(
                        filename
                    )[1]
                    .replace(".", "")
                    .upper()
                ),
                "exists": exists,
                "size_bytes": (
                    os.path.getsize(
                        file_path
                    )
                    if exists
                    else 0
                ),
            }
        )

    return {
        "documents": documents
    }


@app.get("/document/{filename:path}")
def open_document(
    filename: str,
):

    file_path = os.path.join(
        DATA_FOLDER,
        filename,
    )

    data_folder_abs = os.path.abspath(
        DATA_FOLDER
    )

    file_path_abs = os.path.abspath(
        file_path
    )

    try:

        common_path = os.path.commonpath(
            [
                data_folder_abs,
                file_path_abs,
            ]
        )

    except ValueError:

        common_path = ""

    if common_path != data_folder_abs:

        raise HTTPException(
            status_code=400,
            detail="Invalid file path.",
        )

    if not os.path.isfile(
        file_path_abs
    ):

        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    if (
        os.path.splitext(
            file_path_abs
        )[1].lower()
        != ".pdf"
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Only PDF documents can be "
                "opened directly in the browser."
            ),
        )

    def file_iterator():

        with open(
            file_path_abs,
            "rb",
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
            "Content-Disposition": (
                'inline; filename="'
                + os.path.basename(
                    file_path_abs
                )
                + '"'
            )
        },
    )