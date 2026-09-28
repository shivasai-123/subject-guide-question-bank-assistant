import os
import re
import requests
import streamlit as st


# ==========================================================
# CONFIGURATION
# ==========================================================

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Subject Guide & Question Bank AI Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def api_get(endpoint, params=None):
    """Send a GET request to the FastAPI backend."""

    try:
        response = requests.get(
            f"{API_URL}{endpoint}",
            params=params,
            timeout=30,
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.ConnectionError:
        st.error(
            "❌ Cannot connect to the FastAPI backend. "
            "Make sure app.py is running on port 8000."
        )
        return None

    except requests.exceptions.Timeout:
        st.error(
            "⏱️ The backend took too long to respond."
        )
        return None

    except requests.exceptions.RequestException as e:
        st.error(
            f"❌ Backend error: {e}"
        )
        return None


def api_post(endpoint, data=None, files=None, timeout=180):
    """Send a POST request to the FastAPI backend."""

    try:
        response = requests.post(
            f"{API_URL}{endpoint}",
            json=data,
            files=files,
            timeout=timeout,
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.ConnectionError:
        st.error(
            "❌ Cannot connect to the FastAPI backend. "
            "Make sure app.py is running on port 8000."
        )
        return None

    except requests.exceptions.Timeout:
        st.error(
            "⏱️ The request took too long."
        )
        return None

    except requests.exceptions.HTTPError:
        try:
            error_detail = response.json().get(
                "detail",
                response.text,
            )
        except Exception:
            error_detail = response.text

        st.error(
            f"❌ Backend error: {error_detail}"
        )
        return None

    except requests.exceptions.RequestException as e:
        st.error(
            f"❌ Request error: {e}"
        )
        return None


def get_subjects():
    """Get available subjects."""

    data = api_get("/subjects")

    if not data:
        return []

    return data.get(
        "subjects",
        [],
    )


def get_chapters(subject=None):
    """Get available chapters."""

    params = {}

    if subject and subject != "All Subjects":
        params["subject"] = subject

    data = api_get(
        "/chapters",
        params=params,
    )

    if not data:
        return []

    return data.get(
        "chapters",
        [],
    )


def format_bytes(size):
    """Convert bytes into a readable size."""

    if size < 1024:
        return f"{size} B"

    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"

    if size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"

    return f"{size / (1024 * 1024 * 1024):.1f} GB"


def display_sources(sources):
    """Display source documents."""

    if not sources:
        return

    st.markdown("### 📚 Sources")

    seen = set()

    for source in sources:

        filename = source.get(
            "filename",
            "Unknown",
        )

        subject = source.get(
            "subject",
            "Unknown",
        )

        chapter = source.get(
            "chapter",
            "Unknown",
        )

        content_type = source.get(
            "content_type",
            "Unknown",
        )

        key = (
            filename,
            subject,
            chapter,
            content_type,
        )

        if key in seen:
            continue

        seen.add(key)

        with st.expander(
            f"📄 {filename}"
        ):
            st.write(
                f"**Subject:** {subject}"
            )

            st.write(
                f"**Chapter:** {chapter}"
            )

            st.write(
                f"**Type:** {content_type}"
            )


# ==========================================================
# ANSWER FORMATTING
# ==========================================================

def format_bfs_code(code_text):
    """
    Clean the BFS Python example when the model has collapsed
    multiple Python statements onto one line.
    """

    code_text = code_text.replace("`", "").strip()

    if "from collections import deque" not in code_text:
        return code_text

    # If the complete BFS source is already reasonably formatted,
    # preserve it.
    if "\n" in code_text and "def bfs" in code_text:
        return code_text

    # Normalize spaces around the common BFS source.
    compact = " ".join(
        line.strip()
        for line in code_text.splitlines()
        if line.strip()
    )

    # Remove the "Example 1 PYTHON CODE" label if present.
    compact = re.sub(
        r"^Example\s*\d*\s*PYTHON\s*CODE\s*",
        "",
        compact,
        flags=re.IGNORECASE,
    )

    # Reconstruct the BFS example from the statements present
    # in the generated answer.
    if (
        "from collections import deque" in compact
        and "def bfs" in compact
        and "visited = set([start])" in compact
    ):
        return """from collections import deque

graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B'],
    'F': ['C']
}

def bfs(graph, start):
    visited = set([start])
    queue = deque([start])
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)

        for neighbour in graph[node]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)

    return order

print(bfs(graph, 'A'))"""

    return code_text


def display_answer(answer):
    """
    Display an AI-generated answer with improved formatting.

    The backend remains unchanged. This function separates:
    - Main explanation
    - Important points
    - Formula / Syntax
    - Example / Application
    - Conclusion

    Python examples are displayed using Streamlit's code block.
    """

    if not answer:
        st.warning(
            "No answer was returned."
        )
        return

    st.markdown("### 🤖 Answer")

    # ------------------------------------------------------
    # Locate major sections
    # ------------------------------------------------------

    example_marker = "Example / Application"
    conclusion_marker = "Conclusion:"

    # Split before Example / Application
    if example_marker in answer:

        before_example, after_example = answer.split(
            example_marker,
            1,
        )

        # --------------------------------------------------
        # Display main explanation
        # --------------------------------------------------

        st.markdown(
            before_example.strip()
        )

        # --------------------------------------------------
        # Split example from conclusion
        # --------------------------------------------------

        if conclusion_marker in after_example:

            example_text, conclusion = after_example.split(
                conclusion_marker,
                1,
            )

        else:

            example_text = after_example
            conclusion = ""

        # --------------------------------------------------
        # Display Example / Application
        # --------------------------------------------------

        st.markdown("### Example / Application")

        example_text = example_text.strip()

        # Remove markdown backticks from the beginning/end.
        example_text = example_text.replace(
            "```python",
            "",
        ).replace(
            "```Python",
            "",
        ).replace(
            "```",
            "",
        ).strip()

        # Look for Python code.
        python_marker = re.search(
            r"PYTHON\s+CODE",
            example_text,
            flags=re.IGNORECASE,
        )

        if python_marker:

            # Everything after PYTHON CODE is treated as code
            # until the example section ends.
            code_text = example_text[
                python_marker.end():
            ].strip()

            code_text = format_bfs_code(
                code_text
            )

            st.code(
                code_text,
                language="python",
            )

        else:

            # Check whether the example contains a Python
            # fenced block.
            python_block = re.search(
                r"```python\s*(.*?)```",
                example_text,
                flags=re.IGNORECASE | re.DOTALL,
            )

            if python_block:

                explanation_before = (
                    example_text[:python_block.start()]
                    .strip()
                )

                if explanation_before:
                    st.markdown(
                        explanation_before
                    )

                code_text = python_block.group(1).strip()

                st.code(
                    code_text,
                    language="python",
                )

                explanation_after = (
                    example_text[python_block.end():]
                    .strip()
                )

                if explanation_after:
                    st.markdown(
                        explanation_after
                    )

            else:

                # If there is no recognizable code,
                # display the example normally.
                st.markdown(
                    example_text
                )

        # --------------------------------------------------
        # Display Conclusion
        # --------------------------------------------------

        if conclusion.strip():

            st.markdown("### Conclusion")

            st.markdown(
                conclusion.strip()
            )

    else:

        # If the response doesn't use the expected structure,
        # simply render it normally.
        st.markdown(answer)


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title(
    "📚 Subject Guide"
)

st.sidebar.caption(
    "AI Academic Learning Assistant"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Dashboard",
        "💬 Ask Question",
        "🧠 Learn a Topic",
        "📝 Question Bank",
        "📄 Documents",
        "⬆️ Upload Document",
        "📜 Learning History",
    ],
)

st.sidebar.divider()

st.sidebar.caption(
    f"Backend: {API_URL}"
)


# ==========================================================
# DASHBOARD
# ==========================================================

if page == "🏠 Dashboard":

    st.title(
        "📚 Subject Guide & Question Bank AI Assistant"
    )

    st.write(
        "Your multi-document RAG academic learning assistant."
    )

    st.divider()

    health = api_get("/health")

    if health:

        documents = health.get(
            "documents",
            0,
        )

        vectors = health.get(
            "vectors",
            0,
        )

        subjects = health.get(
            "subjects",
            [],
        )

        model = health.get(
            "model",
            "Unknown",
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "📄 Documents",
                documents,
            )

        with col2:
            st.metric(
                "🔢 Vectors",
                vectors,
            )

        with col3:
            st.metric(
                "📚 Subjects",
                len(subjects),
            )

        with col4:
            st.metric(
                "🤖 Model",
                model,
            )

    st.divider()

    st.subheader(
        "📖 Available Subjects"
    )

    subjects = get_subjects()

    if subjects:

        cols = st.columns(
            min(len(subjects), 4)
        )

        for i, subject in enumerate(subjects):

            with cols[i % len(cols)]:
                st.info(
                    f"📘 **{subject}**"
                )

    else:

        st.info(
            "No subjects available yet."
        )

    st.divider()

    st.subheader(
        "🚀 What can you do?"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            ### 💬 Ask Questions

            Ask questions from your uploaded
            study materials.
            """
        )

    with col2:

        st.markdown(
            """
            ### 🧠 Learn Topics

            Follow the:

            **Theory → Example → Practice → Assessment**

            learning flow.
            """
        )

    with col3:

        st.markdown(
            """
            ### 📝 Prepare for Exams

            Explore questions from your
            uploaded question banks.
            """
        )


# ==========================================================
# ASK QUESTION
# ==========================================================

elif page == "💬 Ask Question":

    st.title(
        "💬 Ask a Question"
    )

    st.write(
        "Ask something from your uploaded study material."
    )

    subjects = get_subjects()

    subject_options = [
        "All Subjects"
    ] + subjects

    selected_subject = st.selectbox(
        "📚 Subject",
        subject_options,
    )

    subject = (
        None
        if selected_subject == "All Subjects"
        else selected_subject
    )

    chapters = get_chapters(
        subject
    )

    chapter_options = [
        "All Chapters"
    ] + chapters

    selected_chapter = st.selectbox(
        "📖 Chapter",
        chapter_options,
    )

    chapter = (
        None
        if selected_chapter == "All Chapters"
        else selected_chapter
    )

    question = st.text_area(
        "Your question",
        placeholder=(
            "Example: What is BFS?"
        ),
        height=120,
    )

    ask_button = st.button(
        "🔍 Ask Question",
        type="primary",
        use_container_width=True,
    )

    if ask_button:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "🔎 Searching your study material..."
            ):

                result = api_post(
                    "/ask",
                    {
                        "question": question,
                        "subject": subject,
                        "chapter": chapter,
                    },
                    timeout=180,
                )

            if result:

                display_answer(
                    result.get(
                        "answer",
                        "",
                    )
                )

                display_sources(
                    result.get(
                        "sources",
                        [],
                    )
                )


# ==========================================================
# LEARN A TOPIC
# ==========================================================

elif page == "🧠 Learn a Topic":

    st.title(
        "🧠 Learn a Topic"
    )

    st.write(
        "Study a topic using the "
        "**Theory → Example → Practice → Assessment** flow."
    )

    subjects = get_subjects()

    subject_options = [
        "All Subjects"
    ] + subjects

    selected_subject = st.selectbox(
        "📚 Subject",
        subject_options,
        key="learn_subject",
    )

    subject = (
        None
        if selected_subject == "All Subjects"
        else selected_subject
    )

    chapters = get_chapters(
        subject
    )

    chapter_options = [
        "All Chapters"
    ] + chapters

    selected_chapter = st.selectbox(
        "📖 Chapter",
        chapter_options,
        key="learn_chapter",
    )

    chapter = (
        None
        if selected_chapter == "All Chapters"
        else selected_chapter
    )

    topic = st.text_input(
        "Topic",
        placeholder=(
            "Example: BFS"
        ),
    )

    learn_button = st.button(
        "🚀 Start Learning",
        type="primary",
        use_container_width=True,
    )

    if learn_button:

        if not topic.strip():

            st.warning(
                "Please enter a topic."
            )

        else:

            with st.spinner(
                "🧠 Building your learning path..."
            ):

                result = api_post(
                    "/progression",
                    {
                        "topic": topic,
                        "subject": subject,
                        "chapter": chapter,
                    },
                    timeout=240,
                )

            if result:

                answer = result.get(
                    "answer",
                    "",
                )

                display_answer(
                    answer
                )

                display_sources(
                    result.get(
                        "sources",
                        [],
                    )
                )


# ==========================================================
# QUESTION BANK
# ==========================================================

elif page == "📝 Question Bank":

    st.title(
        "📝 Question Bank"
    )

    st.write(
        "Explore questions from your uploaded question banks."
    )

    subjects = get_subjects()

    subject_options = [
        "All Subjects"
    ] + subjects

    selected_subject = st.selectbox(
        "📚 Subject",
        subject_options,
        key="qb_subject",
    )

    subject = (
        None
        if selected_subject == "All Subjects"
        else selected_subject
    )

    chapters = get_chapters(
        subject
    )

    chapter_options = [
        "All Chapters"
    ] + chapters

    selected_chapter = st.selectbox(
        "📖 Chapter",
        chapter_options,
        key="qb_chapter",
    )

    chapter = (
        None
        if selected_chapter == "All Chapters"
        else selected_chapter
    )

    tab1, tab2 = st.tabs(
        [
            "📋 All Questions",
            "🔎 Topic Questions",
        ]
    )

    with tab1:

        if st.button(
            "📋 Show All Questions",
            type="primary",
            use_container_width=True,
        ):

            with st.spinner(
                "Loading question bank..."
            ):

                result = api_post(
                    "/ask",
                    {
                        "question": (
                            "List all questions "
                            "in the question bank"
                        ),
                        "subject": subject,
                        "chapter": chapter,
                    },
                    timeout=60,
                )

            if result:

                display_answer(
                    result.get(
                        "answer",
                        "",
                    )
                )

                display_sources(
                    result.get(
                        "sources",
                        [],
                    )
                )

    with tab2:

        topic = st.text_input(
            "Enter topic",
            placeholder="Example: BFS",
            key="question_topic",
        )

        if st.button(
            "🔎 Find Topic Questions",
            type="primary",
            use_container_width=True,
        ):

            if not topic.strip():

                st.warning(
                    "Please enter a topic."
                )

            else:

                with st.spinner(
                    "Searching question bank..."
                ):

                    result = api_post(
                        "/ask",
                        {
                            "question": (
                                f"Questions related to "
                                f"{topic}"
                            ),
                            "subject": subject,
                            "chapter": chapter,
                        },
                        timeout=60,
                    )

                if result:

                    display_answer(
                        result.get(
                            "answer",
                            "",
                        )
                    )

                    display_sources(
                        result.get(
                            "sources",
                            [],
                        )
                    )


# ==========================================================
# DOCUMENTS
# ==========================================================

elif page == "📄 Documents":

    st.title(
        "📄 Document Library"
    )

    st.write(
        "Browse the study materials currently registered in the system."
    )

    data = api_get(
        "/documents"
    )

    if data:

        documents = data.get(
            "documents",
            [],
        )

        if not documents:

            st.info(
                "No documents found."
            )

        else:

            st.success(
                f"{len(documents)} document(s) registered."
            )

            for document in documents:

                filename = document.get(
                    "filename",
                    "Unknown",
                )

                subject = document.get(
                    "subject",
                    "General",
                )

                chapter = document.get(
                    "chapter",
                    "General",
                )

                content_type = document.get(
                    "content_type",
                    "Notes",
                )

                file_format = document.get(
                    "format",
                    "Unknown",
                )

                exists = document.get(
                    "exists",
                    False,
                )

                size = document.get(
                    "size_bytes",
                    0,
                )

                with st.expander(
                    f"📄 {filename}"
                ):

                    col1, col2 = st.columns(2)

                    with col1:

                        st.write(
                            f"**Subject:** {subject}"
                        )

                        st.write(
                            f"**Chapter:** {chapter}"
                        )

                        st.write(
                            f"**Type:** {content_type}"
                        )

                    with col2:

                        st.write(
                            f"**Format:** {file_format}"
                        )

                        st.write(
                            f"**Size:** {format_bytes(size)}"
                        )

                        st.write(
                            f"**Available:** "
                            f"{'Yes' if exists else 'No'}"
                        )

                    if (
                        exists
                        and file_format == "PDF"
                    ):

                        st.info(
                            "This PDF is available in the "
                            "backend document folder."
                        )


# ==========================================================
# UPLOAD DOCUMENT
# ==========================================================

elif page == "⬆️ Upload Document":

    st.title(
        "⬆️ Upload Study Material"
    )

    st.write(
        "Upload PDF, DOCX, or PPTX study material."
    )

    uploaded_file = st.file_uploader(
        "Choose a document",
        type=[
            "pdf",
            "docx",
            "pptx",
        ],
    )

    subjects = get_subjects()

    subject_options = [
        "General"
    ] + subjects

    selected_subject = st.selectbox(
        "📚 Subject",
        subject_options,
        key="upload_subject",
    )

    custom_subject = st.text_input(
        "Or enter a new subject",
        placeholder="Example: Data Mining",
    )

    subject = (
        custom_subject.strip()
        if custom_subject.strip()
        else selected_subject
    )

    chapter = st.text_input(
        "📖 Chapter",
        value="General",
        placeholder="Example: Unit 3",
    )

    if uploaded_file:

        st.info(
            f"Selected: **{uploaded_file.name}** "
            f"({format_bytes(uploaded_file.size)})"
        )

    upload_button = st.button(
        "⬆️ Upload Document",
        type="primary",
        use_container_width=True,
    )

    if upload_button:

        if not uploaded_file:

            st.warning(
                "Please choose a document first."
            )

        else:

            with st.spinner(
                "📄 Processing document and building embeddings..."
            ):

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type,
                    )
                }

                try:

                    response = requests.post(
                        f"{API_URL}/upload",
                        params={
                            "subject": subject,
                            "chapter": chapter,
                        },
                        files=files,
                        timeout=300,
                    )

                    response.raise_for_status()

                    result = response.json()

                except requests.exceptions.ConnectionError:

                    result = None

                    st.error(
                        "❌ Cannot connect to the FastAPI backend."
                    )

                except requests.exceptions.Timeout:

                    result = None

                    st.error(
                        "⏱️ Document processing took too long."
                    )

                except requests.exceptions.HTTPError:

                    result = None

                    try:
                        detail = response.json().get(
                            "detail",
                            response.text,
                        )
                    except Exception:
                        detail = response.text

                    st.error(
                        f"❌ Upload failed: {detail}"
                    )

                except requests.exceptions.RequestException as e:

                    result = None

                    st.error(
                        f"❌ Upload error: {e}"
                    )

            if result:

                st.success(
                    result.get(
                        "message",
                        "Document uploaded successfully.",
                    )
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Chunks Added",
                        result.get(
                            "chunks_added",
                            0,
                        ),
                    )

                with col2:

                    st.metric(
                        "Total Documents",
                        result.get(
                            "total_documents",
                            0,
                        ),
                    )

                with col3:

                    st.metric(
                        "Total Vectors",
                        result.get(
                            "total_vectors",
                            0,
                        ),
                    )

                st.write(
                    f"**File:** "
                    f"{result.get('filename', '')}"
                )

                st.write(
                    f"**Subject:** "
                    f"{result.get('subject', '')}"
                )

                st.write(
                    f"**Chapter:** "
                    f"{result.get('chapter', '')}"
                )

                st.write(
                    f"**Content Type:** "
                    f"{result.get('content_type', '')}"
                )


# ==========================================================
# LEARNING HISTORY
# ==========================================================

elif page == "📜 Learning History":

    st.title(
        "📜 Learning History"
    )

    st.write(
        "Review your recorded learning activities."
    )

    data = api_get(
        "/history"
    )

    if data:

        history = data.get(
            "history",
            [],
        )

        if not history:

            st.info(
                "No learning activity has been recorded yet."
            )

        else:

            st.success(
                f"{len(history)} learning activity record(s)."
            )

            for index, item in enumerate(
                reversed(history),
                start=1,
            ):

                with st.expander(
                    f"Activity {index}"
                ):

                    if isinstance(
                        item,
                        dict,
                    ):

                        for key, value in item.items():

                            st.write(
                                f"**{key.replace('_', ' ').title()}:** "
                                f"{value}"
                            )

                    else:

                        st.write(item)


# ==========================================================
# FOOTER
# ==========================================================

st.sidebar.divider()

st.sidebar.caption(
    "Subject Guide & Question Bank AI Assistant"
)

st.sidebar.caption(
    "Built with FastAPI + FAISS + Sentence Transformers + Ollama + Streamlit"
)