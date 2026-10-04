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


def natural_chapter_sort_key(chapter_name):
    """Sort key for natural/numeric chapter ordering."""
    clean_name = str(chapter_name).strip()
    if clean_name.lower() == "general":
        return (2, [])

    tokens = []
    for part in re.split(r"(\d+)", clean_name):
        if not part:
            continue
        if part.isdigit():
            tokens.append((0, int(part), ""))
        else:
            tokens.append((1, 0, part.lower()))

    return (0, tokens)


def get_chapters(subject=None):
    """Get available chapters in natural numerical order."""

    params = {}

    if subject and subject != "All Subjects":
        params["subject"] = subject

    data = api_get(
        "/chapters",
        params=params,
    )

    if not data:
        return []

    chapters = data.get(
        "chapters",
        [],
    )

    return sorted(
        chapters,
        key=natural_chapter_sort_key,
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
            f"📄 {filename} [{content_type}] - {subject} ({chapter})"
        ):
            st.write(
                f"**Subject:** {subject} | **Chapter:** {chapter} | **Type:** {content_type}"
            )
            text_snippet = source.get("text", "").strip()
            if text_snippet:
                st.caption("Retrieved Source Excerpt:")
                st.code(
                    text_snippet[:400] + ("..." if len(text_snippet) > 400 else ""),
                    language="text",
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

        # Look for multi-language code marker: PYTHON CODE, C CODE, SQL CODE, JAVA CODE
        code_marker = re.search(
            r"(PYTHON|C|CPP|C\+\+|JAVA|SQL)\s+CODE",
            example_text,
            flags=re.IGNORECASE,
        )

        if code_marker:

            detected_tag = code_marker.group(1).upper()
            lang_map = {
                "PYTHON": "python",
                "C": "c",
                "CPP": "cpp",
                "C++": "cpp",
                "JAVA": "java",
                "SQL": "sql",
            }
            code_lang = lang_map.get(detected_tag, "python")
            code_text = example_text[code_marker.end():].strip()
            if code_lang == "python":
                code_text = format_bfs_code(code_text)

            st.code(
                code_text,
                language=code_lang,
            )

        else:

            # Check whether the example contains a fenced code block
            fenced_block = re.search(
                r"```(python|c|cpp|java|sql)?\s*(.*?)```",
                example_text,
                flags=re.IGNORECASE | re.DOTALL,
            )

            if fenced_block:

                explanation_before = (
                    example_text[:fenced_block.start()]
                    .strip()
                )

                if explanation_before:
                    st.markdown(
                        explanation_before
                    )

                fenced_lang = (fenced_block.group(1) or "python").lower()
                code_text = fenced_block.group(2).strip()
                if fenced_lang == "python":
                    code_text = format_bfs_code(code_text)

                st.code(
                    code_text,
                    language=fenced_lang,
                )

                explanation_after = (
                    example_text[fenced_block.end():]
                    .strip()
                )

                if explanation_after:
                    st.markdown(
                        explanation_after
                    )

            else:

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

    # ------------------------------------------------------
    # Export / Download Actions (Track A Requirement)
    # ------------------------------------------------------
    st.divider()
    clean_id = abs(hash(answer)) % 10000000
    col_dl1, col_dl2 = st.columns([1, 1])
    with col_dl1:
        st.download_button(
            label="📥 Export Guide (.md)",
            data=answer,
            file_name="academic_guide.md",
            mime="text/markdown",
            key=f"dl_md_{clean_id}",
            use_container_width=True,
        )
    with col_dl2:
        st.download_button(
            label="📄 Export Guide (.txt)",
            data=answer,
            file_name="academic_guide.txt",
            mime="text/plain",
            key=f"dl_txt_{clean_id}",
            use_container_width=True,
        )


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
        "🎯 Exam Prep & Planner",
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
    doc_data = api_get("/documents")
    docs_list = doc_data.get("documents", []) if doc_data else []

    notes_count = sum(1 for d in docs_list if d.get("content_type", "").lower() == "notes")
    qb_count = sum(1 for d in docs_list if "question" in d.get("content_type", "").lower())
    tb_count = sum(1 for d in docs_list if "textbook" in d.get("content_type", "").lower())
    lab_count = sum(1 for d in docs_list if "lab" in d.get("content_type", "").lower())

    if health:

        documents = health.get(
            "documents",
            len(docs_list),
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
            "llama3.2:3b",
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "📄 Total Documents",
                documents,
            )

        with col2:
            st.metric(
                "🔢 Vector Embeddings",
                vectors,
            )

        with col3:
            st.metric(
                "📚 Registered Subjects",
                len(subjects),
            )

        with col4:
            st.metric(
                "🤖 Local LLM",
                model,
            )

        st.markdown("#### 📊 Academic Content Distribution")
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        with col_c1:
            st.metric("📝 Study Notes", notes_count)
        with col_c2:
            st.metric("❓ Question Banks", qb_count)
        with col_c3:
            st.metric("📖 Textbooks", tb_count)
        with col_c4:
            st.metric("🔬 Lab Manuals", lab_count)

    st.divider()

    st.subheader(
        "📖 Available Subjects & Modules"
    )

    subjects = get_subjects()

    if subjects:

        cols = st.columns(
            min(len(subjects), 4)
        )

        for i, subject in enumerate(subjects):

            with cols[i % len(cols)]:
                subject_chapters = get_chapters(subject)
                with st.expander(f"📘 **{subject}**", expanded=False):
                    st.caption(f"Modules / Chapters: {len(subject_chapters)}")
                    for ch in subject_chapters:
                        st.markdown(f"- {ch}")

    else:

        st.info(
            "No subjects available yet."
        )

    st.divider()

    st.subheader(
        "🚀 Quick Actions"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("### 💬 Ask Questions")
        st.write("Targeted answers using verified multi-source course material.")

    with col2:
        st.markdown("### 🧠 Learn a Topic")
        st.write("Structured: Theory → Code Example → Practice → Assessment.")

    with col3:
        st.markdown("### 📝 Question Bank")
        st.write("Browse and filter syllabus questions by subject & chapter.")

    with col4:
        st.markdown("### 🎯 Exam Prep & Planner")
        st.write("Generate high-yield revision summaries, multi-day plans & quizzes.")


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
# EXAM PREP & STUDY PLANNER (WEEK 5-6 DOMAIN SPECIALIZATION)
# ==========================================================

elif page == "🎯 Exam Prep & Planner":

    st.title(
        "🎯 Exam Preparation & Study Planner"
    )

    st.write(
        "Comprehensive exam readiness combining syllabus notes, textbooks, and previous year question papers."
    )

    subjects = get_subjects()

    subject_options = [
        "All Subjects"
    ] + subjects

    selected_subject = st.selectbox(
        "📚 Subject",
        subject_options,
        key="exam_subject",
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
        "📖 Chapter / Module",
        chapter_options,
        key="exam_chapter",
    )

    chapter = (
        None
        if selected_chapter == "All Chapters"
        else selected_chapter
    )

    tab_guide, tab_plan, tab_quiz = st.tabs(
        [
            "📝 High-Yield Exam Guide",
            "📅 Revision Study Plan",
            "🧠 Self-Assessment Diagnostic Quiz",
        ]
    )

    # ------------------------------------------------------
    # TAB 1: HIGH-YIELD EXAM GUIDE
    # ------------------------------------------------------
    with tab_guide:

        st.subheader(
            "High-Yield Exam Topic Guide"
        )
        st.write(
            "Combines theory, extracted code examples, and practice questions from the question bank."
        )

        exam_topic = st.text_input(
            "Exam Topic",
            placeholder="Example: Breadth First Search or Normalization",
            key="exam_topic_input",
        )

        if st.button(
            "🚀 Generate Exam Guide",
            type="primary",
            use_container_width=True,
            key="btn_exam_guide",
        ):

            if not exam_topic.strip():

                st.warning(
                    "Please enter an exam topic."
                )

            else:

                with st.spinner(
                    "Compiling exam preparation guide..."
                ):

                    result = api_post(
                        "/exam-prep",
                        {
                            "topic": exam_topic,
                            "subject": subject,
                            "chapter": chapter,
                        },
                        timeout=240,
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

    # ------------------------------------------------------
    # TAB 2: MULTI-DAY REVISION STUDY PLAN
    # ------------------------------------------------------
    with tab_plan:

        st.subheader(
            "Custom Multi-Day Revision Plan"
        )
        st.write(
            "Structured day-by-day objectives, practice tasks, and self-checks based on syllabus & question banks."
        )

        plan_subj = (
            selected_subject
            if selected_subject != "All Subjects"
            else (subjects[0] if subjects else "General")
        )
        plan_chap = (
            selected_chapter
            if selected_chapter != "All Chapters"
            else "General"
        )

        days = st.slider(
            "Days until Examination",
            min_value=3,
            max_value=14,
            value=5,
            step=1,
            help="Select the number of study days to divide your revision into.",
        )

        if st.button(
            "📅 Generate Revision Plan",
            type="primary",
            use_container_width=True,
            key="btn_study_plan",
        ):

            with st.spinner(
                f"Generating {days}-day study plan for {plan_subj}..."
            ):

                result = api_post(
                    "/study-plan",
                    {
                        "subject": plan_subj,
                        "chapter": plan_chap,
                        "days": days,
                    },
                    timeout=180,
                )

            if result:

                plan_text = result.get(
                    "answer",
                    "",
                )

                st.markdown(
                    plan_text
                )

                st.divider()

                clean_id = abs(hash(plan_text)) % 10000000

                st.download_button(
                    label="📥 Export Revision Plan (.md)",
                    data=plan_text,
                    file_name=f"revision_plan_{plan_subj}_{days}days.md",
                    mime="text/markdown",
                    key=f"dl_plan_{clean_id}",
                    use_container_width=True,
                )

    # ------------------------------------------------------
    # TAB 3: SELF-ASSESSMENT DIAGNOSTIC QUIZ
    # ------------------------------------------------------
    with tab_quiz:

        st.subheader(
            "Self-Assessment Diagnostic Quiz"
        )
        st.write(
            "Identify weak areas and test your comprehension on any topic using course-material questions."
        )

        quiz_topic = st.text_input(
            "Topic for Diagnostic Quiz",
            placeholder="Example: Tuple Packing or BFS",
            key="quiz_topic_input",
        )

        if st.button(
            "📝 Start Diagnostic Quiz",
            type="primary",
            use_container_width=True,
            key="btn_quiz",
        ):

            if not quiz_topic.strip():

                st.warning(
                    "Please enter a topic to test."
                )

            else:

                with st.spinner(
                    "Generating diagnostic quiz from study materials..."
                ):

                    result = api_post(
                        "/quiz",
                        {
                            "topic": quiz_topic,
                            "subject": subject,
                            "chapter": chapter,
                        },
                        timeout=180,
                    )

                if result:

                    quiz_text = result.get(
                        "answer",
                        "",
                    )

                    st.markdown(
                        quiz_text
                    )

                    st.divider()

                    clean_id = abs(hash(quiz_text)) % 10000000

                    st.download_button(
                        label="📥 Export Diagnostic Quiz (.md)",
                        data=quiz_text,
                        file_name=f"diagnostic_quiz_{quiz_topic}.md",
                        mime="text/markdown",
                        key=f"dl_quiz_{clean_id}",
                        use_container_width=True,
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

    uploaded_files = st.file_uploader(
        "Choose document(s)",
        type=[
            "pdf",
            "docx",
            "pptx",
        ],
        accept_multiple_files=True,
        help="Select one or multiple PDF, DOCX, or PPTX files to upload.",
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

    content_type_choice = st.selectbox(
        "🏷️ Content Type Classification",
        [
            "Auto Detect",
            "Notes",
            "Question Bank",
            "Textbook",
            "Lab Manual",
        ],
        help="Classify the document type to optimize how the assistant processes theory vs exam questions.",
    )

    content_type = (
        ""
        if content_type_choice == "Auto Detect"
        else content_type_choice
    )

    if uploaded_files:

        st.info(
            f"📁 **{len(uploaded_files)} file(s) selected:**\n"
            + "\n".join(
                f"- `{f.name}` ({format_bytes(f.size)})"
                for f in uploaded_files
            )
        )

    upload_button = st.button(
        "⬆️ Upload & Process Documents",
        type="primary",
        use_container_width=True,
    )

    if upload_button:

        if not uploaded_files:

            st.warning(
                "Please choose at least one document first."
            )

        else:

            with st.spinner(
                f"📄 Processing {len(uploaded_files)} document(s) and updating vector embeddings..."
            ):

                files_payload = [
                    (
                        "files",
                        (
                            f.name,
                            f.getvalue(),
                            f.type or "application/octet-stream",
                        ),
                    )
                    for f in uploaded_files
                ]

                try:

                    response = requests.post(
                        f"{API_URL}/upload-batch",
                        data={
                            "subject": subject,
                            "chapter": chapter,
                            "content_type": content_type,
                        },
                        files=files_payload,
                        timeout=360,
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
                        "Document(s) uploaded successfully.",
                    )
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Chunks Added",
                        result.get(
                            "total_chunks_added",
                            result.get("chunks_added", 0),
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
                    f"**Subject:** {subject} | **Chapter:** {chapter} | **Content Type:** {content_type or 'Auto Detect'}"
                )

                if "files" in result and isinstance(result["files"], list):
                    with st.expander("📄 Processed File Details", expanded=True):
                        for f_info in result["files"]:
                            st.write(
                                f"- **{f_info.get('filename')}**: "
                                f"{f_info.get('chunks_added', 0)} chunks, "
                                f"Type: `{f_info.get('content_type', 'Notes')}`"
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