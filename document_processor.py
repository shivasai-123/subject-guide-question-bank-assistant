import os
import re
import pymupdf


# ==========================================
# 1. EXTRACT TEXT FROM PDF
# ==========================================

def extract_text_from_pdf(file_path):

    doc = pymupdf.open(file_path)

    full_text = ""

    for page in doc:

        full_text += page.get_text() + "\n"

    doc.close()

    return full_text


# ==========================================
# 2. CLEAN PDF TEXT
# ==========================================

def clean_line(line):

    """
    Remove invisible characters that can appear
    during PDF text extraction.
    """

    if not line:
        return ""

    # Zero-width space
    line = line.replace("\u200b", "")

    # Byte-order mark
    line = line.replace("\ufeff", "")

    # Remove extra spaces
    line = line.strip()

    return line


# ==========================================
# 3. NORMALIZE CHAPTER NAME
# ==========================================

def normalize_chapter_name(chapter):

    """
    Normalize chapter names so that small
    formatting differences do not create
    duplicate chapters.
    """

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


    # Normalize multiple spaces
    title = re.sub(
        r"\s+",
        " ",
        title
    )


    # Remove "in Python" when it is a duplicate
    title = re.sub(
        r"\s+in\s+Python$",
        "",
        title,
        flags=re.IGNORECASE
    )


    # Common formatting corrections
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


    # Remove trailing punctuation
    title = title.rstrip(
        " .:-"
    )


    # Canonical chapter titles
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


    return (
        f"Chapter {number}: {title}"
    )


# ==========================================
# 4. DETECT CHAPTER
# ==========================================

def detect_chapter(
    line,
    document_type
):

    line = clean_line(line)

    if not line:
        return None


    # ======================================
    # PYTHON / CHAPTER STYLE
    # ======================================

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


    # ======================================
    # C PROGRAMMING / SIMPLE NUMBER STYLE
    # ======================================

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
# 5. IDENTIFY DOCUMENT TYPE
# ==========================================

def get_document_type(file_path):

    filename = os.path.basename(
        file_path
    ).lower()


    # ======================================
    # QUESTION BANK
    # ======================================

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


    # ======================================
    # C PROGRAMMING
    # ======================================

    if (
        "c_programming" in filename
        or "c programming" in filename
    ):

        return "simple_number_style"


    # ======================================
    # PYTHON
    # ======================================

    if "python" in filename:

        return "chapter_style"


    # ======================================
    # DEFAULT
    # ======================================

    return "chapter_style"


# ==========================================
# 6. SPLIT DOCUMENT INTO CHAPTERS
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

            lines.append(
                cleaned
            )


    sections = []

    current_chapter = default_chapter

    current_lines = []

    found_first_heading = False


    for line in lines:

        heading = detect_chapter(
            line,
            document_type
        )


        # ----------------------------------
        # New chapter detected
        # ----------------------------------

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

            current_lines = [
                line
            ]

            found_first_heading = True


        # ----------------------------------
        # Normal content
        # ----------------------------------

        else:

            # Ignore text before the first
            # detected chapter.

            if found_first_heading:

                current_lines.append(
                    line
                )


    # --------------------------------------
    # Save final chapter
    # --------------------------------------

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


    # --------------------------------------
    # No chapters found
    # --------------------------------------

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
# 7. CHUNK ONE CHAPTER
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

            chunks.append(
                chunk
            )


        start += step


    return chunks


# ==========================================
# 8. CHUNK QUESTION BANK
# ==========================================

def chunk_question_bank(text):

    """
    Create one chunk for each numbered
    question in a question bank.

    Example:

    1. BFS
    2. DFS
    3. Water Jug

    becomes:

    Chunk 1 -> Question 1
    Chunk 2 -> Question 2
    Chunk 3 -> Question 3
    """

    lines = []

    for line in text.splitlines():

        cleaned = clean_line(line)

        if cleaned:

            lines.append(
                cleaned
            )


    questions = []

    current_question = []


    for line in lines:

        # Detect numbered questions:
        #
        # 1. ...
        # 2. ...
        # 10. ...

        match = re.match(
            r"^\d+\.\s+",
            line
        )


        if match:

            # Save previous question
            if current_question:

                questions.append(
                    " ".join(
                        current_question
                    )
                )


            # Start new question
            current_question = [
                line
            ]


        else:

            # Continue current question
            if current_question:

                current_question.append(
                    line
                )


    # Save final question
    if current_question:

        questions.append(
            " ".join(
                current_question
            )
        )


    return questions


# ==========================================
# 9. PROCESS ONE PDF
# ==========================================

def process_pdf(
    file_path,
    subject="General",
    chapter="General"
):

    text = extract_text_from_pdf(
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
    # QUESTION BANK PROCESSING
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
    # NORMAL DOCUMENT PROCESSING
    # ======================================

    sections = split_into_chapters(
        text=text,

        default_chapter=chapter,

        document_type=document_type
    )


    # Create chunks inside each chapter
    for section in sections:

        chapter_name = (
            normalize_chapter_name(
                section["chapter"]
            )
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
# 10. PROCESS MULTIPLE PDFs
# ==========================================

def process_multiple_pdfs(
    documents
):

    all_documents = []


    for document in documents:

        processed = process_pdf(

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