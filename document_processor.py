import os
import re
import pymupdf
from docx import Document
from pptx import Presentation


# ==========================================
# 1. CLEAN TEXT
# ==========================================

def clean_line(line):

    if not line:
        return ""

    line = line.replace("\u200b", "")
    line = line.replace("\ufeff", "")
    line = line.strip()

    return line


# ==========================================
# 2. EXTRACT TEXT FROM PDF
# ==========================================

def extract_text_from_pdf(file_path):

    doc = pymupdf.open(file_path)

    full_text = ""

    for page in doc:
        full_text += page.get_text() + "\n"

    doc.close()

    return full_text


# ==========================================
# 3. EXTRACT TEXT FROM DOCX
# ==========================================

def extract_text_from_docx(file_path):

    document = Document(file_path)

    lines = []

    for paragraph in document.paragraphs:

        text = clean_line(
            paragraph.text
        )

        if text:
            lines.append(text)

    # Also read tables
    for table in document.tables:

        for row in table.rows:

            row_text = []

            for cell in row.cells:

                text = clean_line(
                    cell.text
                )

                if text:
                    row_text.append(text)

            if row_text:
                lines.append(
                    " | ".join(row_text)
                )

    return "\n".join(lines)


# ==========================================
# 4. EXTRACT TEXT FROM PPTX
# ==========================================

def extract_text_from_pptx(file_path):

    presentation = Presentation(file_path)

    lines = []

    for slide_number, slide in enumerate(
        presentation.slides,
        start=1
    ):

        lines.append(
            f"Slide {slide_number}"
        )

        for shape in slide.shapes:

            if not hasattr(
                shape,
                "text"
            ):
                continue

            text = clean_line(
                shape.text
            )

            if text:
                lines.append(text)

    return "\n".join(lines)


# ==========================================
# 5. EXTRACT TEXT FROM ANY SUPPORTED FILE
# ==========================================

def extract_text_from_file(file_path):

    extension = os.path.splitext(
        file_path
    )[1].lower()

    if extension == ".pdf":

        return extract_text_from_pdf(
            file_path
        )

    if extension == ".docx":

        return extract_text_from_docx(
            file_path
        )

    if extension == ".pptx":

        return extract_text_from_pptx(
            file_path
        )

    raise ValueError(
        "Unsupported file format. "
        "Use PDF, DOCX, or PPTX."
    )


# ==========================================
# 6. NORMALIZE CHAPTER NAME
# ==========================================

def normalize_chapter_name(chapter):

    chapter = clean_line(chapter)

    if not chapter:
        return "General"

    match = re.match(
        r"^Chapter\s*(\d+)\s*[:.]?\s*(.+)$",
        chapter,
        re.IGNORECASE
    )

    if not match:
        return chapter

    number = int(
        match.group(1)
    )

    title = match.group(2).strip()

    title = re.sub(
        r"\s+",
        " ",
        title
    )

    title = re.sub(
        r"\s+in\s+Python$",
        "",
        title,
        flags=re.IGNORECASE
    )

    title = title.replace(
        "ControlFlow",
        "Control Flow"
    )

    title = re.sub(
        r"Data\s+Structuress+",
        "Data Structures",
        title,
        flags=re.IGNORECASE
    )

    title = title.rstrip(
        " .:-"
    )

    canonical_titles = {

        1: "Introduction to Python",
        2: "Environment Setup",
        3: "Python Syntax Basics",
        4: "Data Types",
        5: "Operators",
        6: "String Operations",
        7: "Control Flow",
        8: "Loops and Iteration",
        9: "Data Structures",
        10: "Functions",
        11: "Advanced Functions",
        12: "Scope and Namespaces",
        13: "Modules and Packages",
        14: "Object-Oriented Programming",
        15: "Advanced OOP",
        16: "File Handling and Error Management",
        17: "Advanced Python Concepts",
        18: "Concurrent and Asynchronous Programming"
    }

    if number in canonical_titles:
        title = canonical_titles[number]

    return f"Chapter {number}: {title}"


# ==========================================
# 7. DOCUMENT TYPE
# ==========================================

def get_document_type(file_path):

    filename = os.path.basename(
        file_path
    ).lower()

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

    for keyword in question_bank_keywords:

        if keyword in filename:
            return "question_bank"

    if (
        "c_programming" in filename
        or "c programming" in filename
    ):
        return "simple_number_style"

    if "python" in filename:
        return "chapter_style"

    return "general"


# ==========================================
# 8. DETECT CHAPTER
# ==========================================

def detect_chapter(
    line,
    document_type
):

    line = clean_line(line)

    if not line:
        return None

    if document_type == "chapter_style":

        match = re.match(
            r"^Chapter\s+\d+\s*[:.]?\s*.+$",
            line,
            re.IGNORECASE
        )

        if match:

            return normalize_chapter_name(
                line
            )

        return None

    if document_type == "simple_number_style":

        match = re.match(
            r"^\d+\.\s+[A-Za-z][A-Za-z0-9 &()/\\*+\-'.:]*$",
            line
        )

        if match:
            return line

        return None

    return None


# ==========================================
# 9. SPLIT INTO CHAPTERS
# ==========================================

def split_into_chapters(
    text,
    default_chapter="General",
    document_type="chapter_style"
):

    lines = []

    for line in text.splitlines():

        cleaned = clean_line(line)

        if cleaned:
            lines.append(cleaned)

    sections = []

    current_chapter = default_chapter
    current_lines = []

    found_first_heading = False

    for line in lines:

        heading = detect_chapter(
            line,
            document_type
        )

        if heading:

            if (
                found_first_heading
                and current_lines
            ):

                sections.append(
                    {
                        "chapter":
                            current_chapter,

                        "lines":
                            current_lines
                    }
                )

            current_chapter = heading

            current_lines = [line]

            found_first_heading = True

        else:

            if found_first_heading:
                current_lines.append(line)

    if (
        found_first_heading
        and current_lines
    ):

        sections.append(
            {
                "chapter":
                    current_chapter,

                "lines":
                    current_lines
            }
        )

    if not sections:

        sections.append(
            {
                "chapter":
                    default_chapter,

                "lines":
                    lines
            }
        )

    return sections


# ==========================================
# 10. NORMAL CHUNKING
# ==========================================

def chunk_chapter(
    lines,
    chunk_size=8,
    overlap=2
):

    chunks = []

    start = 0

    step = chunk_size - overlap

    while start < len(lines):

        end = start + chunk_size

        chunk_lines = lines[
            start:end
        ]

        chunk = "\n".join(
            chunk_lines
        ).strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks


# ==========================================
# 11. QUESTION BANK CHUNKING
# ==========================================

def chunk_question_bank(text):

    lines = []

    for line in text.splitlines():

        cleaned = clean_line(line)

        if cleaned:
            lines.append(cleaned)

    questions = []

    current_question = []

    for line in lines:

        match = re.match(
            r"^\d+\.\s+",
            line
        )

        if match:

            if current_question:

                questions.append(
                    " ".join(
                        current_question
                    )
                )

            current_question = [
                line
            ]

        else:

            if current_question:

                current_question.append(
                    line
                )

    if current_question:

        questions.append(
            " ".join(
                current_question
            )
        )

    return questions


# ==========================================
# 12. PROCESS ONE FILE
# ==========================================

def process_pdf(
    file_path,
    subject="General",
    chapter="General"
):

    return process_file(
        file_path,
        subject,
        chapter
    )


def process_file(
    file_path,
    subject="General",
    chapter="General"
):

    text = extract_text_from_file(
        file_path
    )

    filename = os.path.basename(
        file_path
    )

    document_type = get_document_type(
        file_path
    )

    documents = []


    # ======================================
    # QUESTION BANK
    # ======================================

    if document_type == "question_bank":

        questions = chunk_question_bank(
            text
        )

        for question in questions:

            documents.append(
                {
                    "text":
                        question,

                    "metadata": {

                        "filename":
                            filename,

                        "subject":
                            subject,

                        "chapter":
                            chapter,

                        "content_type":
                            "Question Bank"
                    }
                }
            )

        return documents


    # ======================================
    # NORMAL DOCUMENT
    # ======================================

    sections = split_into_chapters(
        text=text,
        default_chapter=chapter,
        document_type=document_type
    )

    for section in sections:

        if document_type == "chapter_style":

            chapter_name = (
                normalize_chapter_name(
                    section["chapter"]
                )
            )

        else:

            chapter_name = (
                section["chapter"]
            )


        chunks = chunk_chapter(
            section["lines"],
            chunk_size=8,
            overlap=2
        )


        for chunk in chunks:

            documents.append(
                {
                    "text":
                        chunk,

                    "metadata": {

                        "filename":
                            filename,

                        "subject":
                            subject,

                        "chapter":
                            chapter_name
                    }
                }
            )


    return documents


# ==========================================
# 13. PROCESS MULTIPLE FILES
# ==========================================

def process_multiple_pdfs(
    documents
):

    all_documents = []

    for document in documents:

        processed = process_file(

            file_path=document[
                "file_path"
            ],

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