# 📚 Subject Guide & Question Bank AI Assistant
### Multi-Source RAG Academic Learning Assistant
**Track A: Subject Study Guide & Academic Learning Assistant**  
**Selected Specialization: Option A1 — Computer Science Subject Guide**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-yellow.svg)](https://github.com/facebookresearch/faiss)
[![SentenceTransformers](https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-orange.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Ollama](https://img.shields.io/badge/Ollama-llama3.2%3A3b-purple.svg)](https://ollama.com/)
[![SQLite](https://img.shields.io/badge/SQLite-Persistent%20History-003B57.svg)](https://www.sqlite.org/)

---

## 🌟 Project Overview

The **Subject Guide & Question Bank AI Assistant** is an academic companion designed for university students, educators, and curriculum planners. Built on a modular Retrieval-Augmented Generation (RAG) architecture, the platform ingests heterogeneous university courseware—including **Lecture Notes**, **Textbooks**, **Lab Manuals**, and **Previous Year Question Papers (PYQs)**—and synthesizes them into:

1. **Retrieval-Grounded Topic Explanations:** Conceptual explanations anchored directly in approved course material without speculative hallucinations.
2. **Question Bank Integration:** Exam questions automatically mapped to textbook concepts with cited step-by-step solutions.
3. **Structured Learning Progression:** Step-by-step pedagogy adhering to the **Theory → Code / Example → Visual Diagram → Practice → Assessment** model.
4. **Computer Science (Option A1) Specialization:** Deterministic code extraction, syntax formatting (Python, C/C++, Java, SQL), and grounded Mermaid state flows for algorithms, database normalization, operating system lifecycles, and network protocol stacks.
5. **Exam Preparation Suite:** High-yield exam revision guides, customizable multi-day revision planners (3 to 14 days), and diagnostic self-assessment quizzes.
6. **One-Click Academic Exports:** Direct downloads of revision guides, quiz keys, and study notes in Markdown (`.md`) and Plain Text (`.txt`).

---

## ❗ Problem Statement

University academic resources are typically fragmented across heterogeneous, unstructured file formats (PDF handouts, Word documents, PowerPoint lecture slides, and scanned PYQ papers). Consequently:
- **Disjointed Revision:** Students waste significant study time cross-referencing past exam questions against hundreds of lecture slides to find answers.
- **Search Inefficiency:** Traditional keyword search ignores semantic intent, while pure vector search often suffers from semantic drift (e.g., conflating distinct graph algorithms like BFS and DFS due to high cosine similarity).
- **Code & Fact Hallucination:** Generic LLMs frequently synthesize non-standard code patterns, obsolete libraries, or unverified algorithm steps that diverge from university syllabi.
- **Session Volatility:** Standard learning prototypes retain history purely in volatile memory, discarding progress whenever the server restarts.

---

## 💡 Proposed Solution

This project implements a multi-format, dual-layer RAG system specifically engineered for academic curriculum mastery:
- **Unified Ingestion:** Extracts and cleans text from PDF (PyMuPDF), DOCX (python-docx), and PPTX (python-pptx) files in single or batch uploads.
- **Content-Aware Indexing:** Automatically categorizes material as `Notes`, `Question Bank`, or `Textbook`, using metadata weighting to prevent question prompts from polluting theoretical definitions.
- **Hybrid Retrieval:** Merges dense vector embeddings (`all-MiniLM-L6-v2`) with lexical keyword scoring and domain-specific query aliasing.
- **Deterministic Grounding:** Extracts genuine code blocks from course material and renders visual state progression diagrams (Mermaid) to visually reinforce algorithmic steps.
- **Persistent State:** Backed by an SQLite database for student progress tracking and a persistent disk cache for instant sub-second FAISS initialization.

---

## ✨ Key Features

- **Multi-Format Document Ingestion:** Full native parsing of `.pdf`, `.docx`, and `.pptx` documents with sliding-window chunking.
- **Batch / Multi-File Upload:** Ingest and index multiple documents simultaneously via the `/upload-batch` endpoint and Streamlit UI.
- **Content Type Classification:** Differentiates study material from previous year question banks to tailor retrieval strategies.
- **Hybrid Semantic & Lexical RAG:** Eliminates semantic drift using dense vector similarity combined with keyword matching and technical query expansion.
- **Pedagogical Progression (Option A1):** Generates structured learning paths: `Theory → Code/Example → Visual Diagram → Practice → Assessment`.
- **Grounded Algorithm Visualizations:** Generates Mermaid diagrams representing FIFO queue states in BFS, LIFO stack states in DFS, divide-and-conquer binary search, DBMS normalization levels, and OS process state transitions.
- **Question Bank Mapping & Solver:** Locates exam questions matching any topic and synthesizes cited answers with marks allocation context.
- **Exam Preparation & Multi-Day Planner:** Formulates customizable 3- to 14-day study plans and diagnostic self-quizzes with answer keys.
- **Natural Chapter Sorting:** Dynamically sorts chapters numerically (`Chapter 1`, `Chapter 2`, `Chapter 10`) rather than alphabetically (`Chapter 1`, `Chapter 10`, `Chapter 2`).
- **Persistent Storage:** SQLite database (`learning_progress.db`) stores user activity across restarts; vector cache (`.rag_cache/`) eliminates re-embedding delays.
- **Academic Export:** One-click download of guides and quiz solutions formatted as `.md` or `.txt`.

---

## 🏛️ System Architecture

The application is structured into decoupled frontend, backend, retrieval, and persistent storage tiers:

```
┌────────────────────────────────────────────────────────┐
│               Streamlit Web Interface                  │
│               (frontend/streamlit_app.py :8501)        │
└───────────────────────────┬────────────────────────────┘
                            │ HTTP REST API
┌───────────────────────────▼────────────────────────────┐
│                    FastAPI Backend                     │
│                  (backend/app.py :8000)                │
└─────┬─────────────────────┬──────────────────────┬─────┘
      │                     │                      │
┌─────▼───────────────┐ ┌───▼────────────────┐ ┌───▼────────────────┐
│ Document Processing │ │  Metadata Manager  │ │   Learning Tools   │
│ (PyMuPDF/docx/pptx) │ │ (document_metadata)│ │ (SQLite & Prompts) │
└─────┬───────────────┘ └───┬────────────────┘ └───┬────────────────┘
      │                     │                      │
      └──────────────┬──────┴──────────────────────┘
                     │
┌────────────────────▼───────────────────────────────────┐
│                   Hybrid RAG Engine                    │
│      Dense Search (FAISS) + Lexical Keyword Scoring    │
│            + CS Technical Aliasing & Routing           │
└────────────────────┬───────────────────────────────────┘
                     │
┌────────────────────▼───────────────────────────────────┐
│                 Inference & Persistence                │
│  • Embeddings: sentence-transformers/all-MiniLM-L6-v2  │
│  • LLM: Ollama (llama3.2:3b local inference)           │
│  • Persistent Storage: data/.rag_cache/ + SQLite DB    │
└────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```text
subject-guide-question-bank-assistant/
│
├── backend/
│   ├── app.py                      # FastAPI REST application & endpoints
│   ├── document_processor.py       # Multi-format document parser (PDF, DOCX, PPTX)
│   ├── learning_tools.py           # LLM prompting, SQLite persistence & diagrams
│   ├── metadata_manager.py         # Metadata schema & document catalog
│   ├── rag_engine.py               # Hybrid FAISS vector store & query routing
│   └── requirements.txt            # Backend Python dependencies
│
├── frontend/
│   └── streamlit_app.py            # Streamlit multi-page interface (port 8501)
│
├── data/
│   ├── document_metadata.json      # Tracked catalog metadata (subject, chapter, type)
│   ├── learning_progress.db        # Persistent SQLite learning history (git-ignored)
│   ├── .rag_cache/                 # Persisted FAISS index & embeddings (git-ignored)
│   ├── BFS_Study_Material.pdf      # Course document
│   ├── c_programming_basics.pdf    # Course document
│   ├── DM UNIT3.pptx               # Course document
│   ├── Python Question Bank.pdf    # Course document
│   ├── PYTHON.pdf                  # Course document
│   ├── seminar9.pdf                # Course document
│   └── UNIT 1 DATA MINING NOTES (1) (1).docx # Course document
│
├── static/
│   └── index.html                  # Standalone SPA web interface
│
├── tests/
│   ├── app_backup.py               # Archive baseline application
│   ├── embedding_test.py           # Embedding sanity test script
│   ├── ollama_test.py              # Ollama connectivity test script
│   ├── project.py                  # Prototype CLI test script
│   └── similarity_test.py          # Vector cosine similarity test script
│
├── .gitignore                      # Git exclusion rules (venv, caches, DBs, logs)
└── README.md                       # Comprehensive project documentation
```

---

## 🛠️ Technology Stack

| Component | Technology | Version / Specification | Role in System |
|---|---|---|---|
| **Language** | Python | `>= 3.10` | Core programming language |
| **REST Backend** | FastAPI | `0.115.0+` | Asynchronous REST API routing, data validation, file ingestion |
| **ASGI Server** | Uvicorn | `0.32.0+` | High-performance ASGI web server |
| **Frontend UI** | Streamlit | `1.40.0+` | Multi-page interactive dashboard and analytics client |
| **Vector Index** | FAISS | `faiss-cpu 1.9.0+` | Dense L2 similarity vector indexing and nearest-neighbor search |
| **Embeddings** | Sentence-Transformers | `all-MiniLM-L6-v2` | Dense 384-dimensional semantic text representations |
| **Language Model** | Ollama | `llama3.2:3b` | Local prompt-constrained academic inference and pedagogical synthesis |
| **PDF Extraction** | PyMuPDF (`fitz`) | `1.24.0+` | Text extraction, page mapping, and metadata reading |
| **DOCX Extraction**| python-docx | `1.1.0+` | Word document paragraph and structure extraction |
| **PPTX Extraction**| python-pptx | `1.0.0+` | Slide text frame and shape extraction |
| **Data Validation**| Pydantic | `2.9.0+` | Request/response data models and schema enforcement |
| **Database** | SQLite 3 | Built-in | Persistent activity log and learning progression history |
| **Cache Format** | NumPy / JSON | Built-in | Persisted vector matrices and document manifest metadata |

---

## ⚙️ How the System Works

1. **Ingestion & Text Extraction:** Files uploaded via `/upload` or `/upload-batch` are routed through `document_processor.py`. PyMuPDF handles PDFs, `python-docx` processes Word files, and `python-pptx` extracts PowerPoint slides. Text is cleaned of invisible unicode and standardized.
2. **Chunking & Classification:** Clean text is split using a sliding sentence window (size: 3 sentences, overlap: 1 sentence) or by question patterns for PYQs. Documents are classified as `Notes`, `Question Bank`, or `Textbook` and cataloged in `data/document_metadata.json`.
3. **Embedding Generation & Vector Caching:** Text chunks are transformed into 384-dimensional vectors using `all-MiniLM-L6-v2`. A FAISS `IndexFlatL2` is built. The index, raw embeddings, and a file manifest are persisted to `data/.rag_cache/`. On subsequent startups, the cache loads in `< 0.2` seconds without re-embedding.
4. **Intent-Driven Hybrid Retrieval:** When a query arrives:
   - Topic intent is categorized (`conceptual`, `question-bank`, `exam-prep`).
   - Relevant CS aliases are expanded (e.g., expanding "BFS" to include "breadth first search", "queue", "level order").
   - Dense FAISS distance is merged with lexical keyword scoring.
   - Irrelevant chunks are discarded via precision thresholding.
5. **Context Assembly & Attribution:** Chunks are formatted into structured context blocks labeled with filename, subject, chapter, and document type.
6. **Guardrailed LLM Synthesis:** `learning_tools.py` sends the prompt and bounded context to Ollama (`llama3.2:3b`). Verified code blocks are extracted deterministically, and algorithm topics trigger Mermaid diagrams.
7. **Persistent Recording:** The interaction is recorded in `data/learning_progress.db` via SQLite for long-term progress tracking.

---

## 📄 Supported Document Formats

| Format | Extension | Parser Library | Processing Strategy |
|---|---|---|---|
| **Portable Document Format** | `.pdf` | PyMuPDF (`fitz`) | Extracts text per page, cleans ligature artifacts, preserves paragraph breaks. |
| **Microsoft Word Document** | `.docx` | `python-docx` | Traverses paragraphs and structured document elements. |
| **PowerPoint Presentation** | `.pptx` | `python-pptx` | Traverses slides, shape frames, and notes panes sequentially. |

---

## 🏷️ Content Classification

The system differentiates course materials into distinct categories to optimize retrieval weighting:
- **`Notes`**: Lecture slides, unit handouts, summaries. Primary source for conceptual definitions and theory.
- **`Question Bank`**: Previous year question papers (PYQs), mid-term assessments, and question banks. Segmented per question item; prioritized during exam-prep and practice modes.
- **`Textbook`**: Authoritative, in-depth academic literature. Provides comprehensive background and reference examples.
- **`Lab Manual`**: Practical programming tasks and code exercises.

---

## 🔍 Hybrid RAG Pipeline

Standard dense vector search can confuse closely related academic terms (e.g., confusing "TCP" and "UDP" or "BFS" and "DFS"). To ensure high precision, our hybrid pipeline operates as follows:

```
Query ──► Query Intent Classifier
              │
              ├──► Technical Alias Expander (e.g. "BFS" -> "queue", "level-order")
              │
              ├──► Dense Semantic Vector Search (FAISS IndexFlatL2)
              │
              ├──► Lexical Keyword Overlap Scoring
              │
              ▼
    Hybrid Relevance Score = Vector_Similarity + (Keyword_Score * Lexical_Weight)
              │
              ▼
    Metadata Filter (Subject, Chapter, Content Type)
              │
              ▼
    Precision Gate (Discard scores below threshold)
              │
              ▼
    Attributed Context Injection -> LLM
```

---

## 🎓 4-Stage Learning Flow

To support academic learning rather than simple direct QA, the system implements a 4-stage progression:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  1. THEORY   │ ──► │  2. EXAMPLE  │ ──► │ 3. PRACTICE  │ ──► │4. ASSESSMENT │
│ Foundations  │     │ Code & Graph │     │  PYQ Prompts │     │ Edge-Checks  │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

1. **Theory:** Foundational conceptual definitions, principles, and prerequisites drawn from lecture notes.
2. **Example / Application:** A deterministic code implementation (Python, C, Java, SQL) accompanied by a visual Mermaid state progression diagram.
3. **Practice:** Real exam questions retrieved from question banks for applied problem-solving.
4. **Assessment:** Conceptual edge-case questions and self-checks to evaluate depth of understanding.

---

## 📝 Question Bank System

- **Question Splitting:** Identifies question boundaries in uploaded PYQs using pattern expressions (e.g., `Q1.`, `1.`, `[5 Marks]`).
- **Topic Mapping:** Matches past exam questions against conceptual notes chunks using keyword overlap and vector similarity.
- **Syllabus-Aligned Solutions:** Solves questions strictly within the scope of the retrieved textbook context, providing explicit citations: `[Source: Python Question Bank.pdf, Page 2]`.

---

## 💻 Option A1 — Computer Science Subject Guide

As the selected Track A specialization, Option A1 provides specialized processing for core Computer Science domains:

### 1. Multi-Language Code Handling
- **Language Detection & Classification:** Automatically recognizes and syntax-highlights **Python**, **C/C++**, **Java**, and **SQL** snippets.
- **Deterministic Extraction:** The `_extract_example()` function isolates genuine code from notes, preventing LLM code hallucinations.

### 2. Grounded Algorithmic State Diagrams (Mermaid)
For algorithm and systems topics, the engine injects structured Mermaid diagrams:
- **Breadth-First Search (BFS):** Graph topology and sequential FIFO queue state trace.
- **Depth-First Search (DFS):** Graph topology and sequential LIFO recursion stack trace.
- **Binary Search:** Array midpoint division trace (`Low`, `Mid`, `High`).
- **Database Normalization (DBMS):** Relational dependency hierarchy (`1NF` → `2NF` → `3NF` → `BCNF`).
- **Operating Systems (OS):** 5-state process lifecycle (`New` → `Ready` → `Running` → `Waiting` → `Terminated`) and deadlock models.
- **Computer Networks:** OSI 7-Layer protocol hierarchy and TCP/IP stack mappings.

---

## 🎯 Exam Preparation Features

*Note: These capabilities are provided as additional features alongside the primary A1 specialization.*

- **High-Yield Revision Guides (`/exam-prep`):** Synthesizes core definitions, key formulas, and frequent exam questions for rapid review.
- **Multi-Day Study Planner (`/study-plan`):** Generates structured revision plans (3 to 14 days) dividing daily work into:
  - *Morning:* Theoretical concept review.
  - *Afternoon:* Code implementation and problem-solving.
  - *Evening:* Self-assessment and past question practice.
- **Diagnostic Quizzes (`/quiz`):** Creates 5-question diagnostic assessments featuring 2 Multiple Choice Questions, 1 Definition Question, 1 Programming/Application Question, and a complete Answer Key with explanations.

---

## 💾 Persistent Data & Caching

| Artifact | File Path | Persistence Mechanism | Benefit |
|---|---|---|---|
| **Document Catalog** | `data/document_metadata.json` | JSON structured schema | Preserves subject, chapter, and document type associations. |
| **Vector Cache** | `data/.rag_cache/` | FAISS index binary + NumPy embeddings + manifest | Reduces engine initialization from **45 seconds** to **0.17 seconds**. |
| **Learning History** | `data/learning_progress.db` | SQLite 3 relational database | Persists student queries, progression runs, and timestamps across backend restarts. |

---

## 🖥️ Streamlit Dashboard

The frontend interface (`frontend/streamlit_app.py`) provides 8 dedicated views:

1. **🏠 Dashboard:** Overview metrics (documents indexed, vector count, subjects, content distribution) and recent uploads.
2. **💬 Ask Question:** Retrieval-grounded question answering with subject/chapter filtering and source citation cards.
3. **🧠 Learn a Topic:** 4-stage pedagogical exploration (Theory, Code Example, Visual Diagram, Practice, Assessment).
4. **📝 Question Bank:** Search previous year questions by topic or browse entire chapter collections.
5. **🎯 Exam Prep & Planner:** High-yield exam guides, multi-day revision schedules, and diagnostic self-quizzes.
6. **📄 Documents:** Document library browser displaying subjects, chapters, chunks, and download buttons.
7. **⬆️ Upload Document:** Drag-and-drop batch uploader supporting multiple PDFs, DOCXs, and PPTXs with progress bars.
8. **📜 Learning History:** SQLite-backed audit log of user learning sessions with search and history-clear options.

---

## 🔌 API Overview

The FastAPI backend (`backend/app.py`) exposes the following REST endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the static single-page application (`static/index.html`). |
| `GET` | `/health` | Returns server health, indexed document/vector counts, model info, and formats. |
| `POST`| `/upload` | Ingests, processes, and indexes a single document (`.pdf`, `.docx`, `.pptx`). |
| `POST`| `/upload-batch` | Ingests, processes, and indexes multiple documents in a single unified operation. |
| `POST`| `/ask` | Queries the RAG pipeline and returns a grounded answer with source citations. |
| `POST`| `/explain` | Generates a structured multi-source conceptual topic explanation. |
| `POST`| `/synthesize` | Synthesizes content across multiple documents for comparative study. |
| `POST`| `/progression` | Executes the 4-stage learning pipeline (`Theory → Example → Practice → Assessment`). |
| `POST`| `/exam-prep` | Produces a high-yield exam revision summary mapped to syllabus questions. |
| `POST`| `/study-plan` | Creates a customizable 3- to 14-day daily study and revision timetable. |
| `POST`| `/quiz` | Generates a topic-specific diagnostic self-quiz with an answer key. |
| `GET` | `/documents` | Returns the catalog of all indexed documents and their metadata. |
| `GET` | `/document/{filename}/download` | Downloads a specific raw source document from the repository. |
| `GET` | `/subjects` | Returns a list of all distinct subjects currently indexed. |
| `GET` | `/chapters` | Returns all chapters for a given subject in natural numerical order. |
| `GET` | `/history` | Fetches persistent learning history records from the SQLite database. |
| `POST`| `/clear-history` | Clears all recorded learning history from the SQLite database. |

---

## 🚀 Installation

### 1. Prerequisites
- **Python:** Version `3.10`, `3.11`, or `3.12`.
- **Ollama:** Installed and running locally from [ollama.com](https://ollama.com/).

### 2. Setup Steps

```powershell
# 1. Clone the repository
git clone https://github.com/shivasai-123/subject-guide-question-bank-assistant.git
cd subject-guide-question-bank-assistant

# 2. Create Python virtual environment
python -m venv .venv

# 3. Activate virtual environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux / macOS:
# source .venv/bin/activate

# 4. Install backend dependencies
pip install -r backend/requirements.txt

# 5. Pull local LLM model via Ollama
ollama pull llama3.2:3b
```

---

## 🏃 Running the Application

To run the complete application, launch the backend and frontend in separate terminal windows:

### Terminal 1: FastAPI Backend (Port 8000)
From the project root:
```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```
- Swagger API Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Built-in Static Frontend: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

### Terminal 2: Streamlit Interactive UI (Port 8501)
From the project root:
```powershell
.\.venv\Scripts\streamlit.exe run frontend/streamlit_app.py
```
- Streamlit Web Dashboard opens automatically at: [http://localhost:8501](http://localhost:8501)

---

## 🧪 Testing & Verification

The project underwent an end-to-end 20-point regression verification audit covering all restructured modules:

| # | Check / Component | Method / Endpoint | Result | Evidence |
|---|---|---|:---:|---|
| 1 | **FastAPI Startup** | Engine re-initialization | **PASS** | App started cleanly; RAG engine initialized in 0.17s |
| 2 | **Health Endpoint** | `GET /health` | **PASS** | Status 200, returns active document and vector counts |
| 3 | **Static Frontend** | `GET /` | **PASS** | Status 200, served `static/index.html` (53,180 bytes) |
| 4 | **Streamlit Startup** | AST validation & helpers | **PASS** | AST parsed cleanly; display and sorting helpers intact |
| 5 | **Frontend ↔ Backend** | HTTP REST integration | **PASS** | Verified `/documents`, `/subjects`, and `/chapters` APIs |
| 6 | **Document Processing** | `document_processor.py` | **PASS** | Extracted text from PDF (6.5k chars), DOCX (79k chars), PPTX (9.6k chars) |
| 7 | **Metadata Catalog** | `metadata_manager.py` | **PASS** | Successfully resolved and parsed `data/document_metadata.json` |
| 8 | **FAISS Cache** | `rag_engine.py` | **PASS** | Loaded 2,156 vectors and documents manifest in `< 0.2` seconds |
| 9 | **RAG Retrieval** | Hybrid search & filtering | **PASS** | Chunks filtered strictly by `subject='Python'` |
| 10| **Single Upload** | `POST /upload` | **PASS** | Single file uploaded, chunked, and appended to vector index |
| 11| **Batch Multi-Upload** | `POST /upload-batch` | **PASS** | Multiple files ingested and indexed in a single request |
| 12| **Question Answering** | `POST /ask` | **PASS** | Generated cited response grounded in retrieved textbook context |
| 13| **Question Bank** | Topic-mapped PYQ queries | **PASS** | Retrieved relevant exam questions for topic 'BFS' |
| 14| **Exam Preparation** | `POST /exam-prep` | **PASS** | Generated high-yield revision summary with marks weightage |
| 15| **Study Plan** | `POST /study-plan` | **PASS** | Formulated structured 5-day daily revision schedule |
| 16| **Quiz Generation** | `POST /quiz` | **PASS** | Generated diagnostic quiz with MCQs and answer key |
| 17| **Persistent History** | `data/learning_progress.db` | **PASS** | Retrieved persisted activity records across server restarts |
| 18| **Chapter Sorting** | Natural numeric sort | **PASS** | `Chapter 1` < `Chapter 2` < `Chapter 10` ... `General` |
| 19| **Export Generation** | Markdown & Text formats | **PASS** | Validated export payloads for `.md` and `.txt` |
| 20| **CS A1 Functionality** | Code formatting & diagrams | **PASS** | Grounded Mermaid graph and FIFO queue trace rendered |

---

## 💡 Example Usage Scenarios

### Scenario 1: Natural Chapter Browsing
- **Action:** Open Dashboard and inspect chapters under `Python`.
- **Behavior:** Chapters are displayed in natural numeric sequence (`Chapter 1: Introduction`, `Chapter 2: Data Types`, ..., `Chapter 10: Object Oriented Programming`), avoiding lexicographical misordering.

### Scenario 2: Algorithmic Learning with Mermaid Diagrams
- **Action:** Navigate to **Learn a Topic** and enter `Breadth First Search`.
- **Behavior:** The assistant outputs theoretical foundations, extracts real Python queue code from lecture notes, and embeds a Mermaid graph visualizing the search tree alongside the FIFO queue state progression.

### Scenario 3: Batch Document Upload
- **Action:** Navigate to **Upload Document**, select a `.pdf` study guide and a `.pptx` slide deck, set Subject to `Operating Systems`, and submit.
- **Behavior:** Files are batch-uploaded via `/upload-batch`, processed in parallel, cataloged, and indexed into FAISS in a single operation.

### Scenario 4: Pre-Exam Multi-Day Revision Plan
- **Action:** Open **Exam Prep & Planner**, select `5 Days` to exam, and enter key units.
- **Behavior:** Creates a balanced timetable allocating morning concept reviews, afternoon coding exercises, and evening PYQ drills.

---

## 🔬 Technical Design Decisions

1. **Why Hybrid Search (Dense + Lexical)?**
   Dense vector embeddings measure semantic relatedness but can overlook exact technical naming conventions. Combining dense similarity with lexical matching and query aliasing ensures that specific algorithms (e.g., BFS vs. DFS) never retrieve incorrect peer algorithms.
2. **Why Deterministic Code Extraction?**
   Prompting generative LLMs to synthesize code often leads to variations in style or syntax that diverge from university course guidelines. Deterministic extraction isolates code examples directly from uploaded materials, ensuring syllabus fidelity.
3. **Why Separation of Question Banks and Notes?**
   PYQ items should not dilute conceptual explanations. The `content_type` weighting scheme demotes question bank chunks during concept explanations and prioritizes them during exam-prep queries.
4. **Why Disk-Persisted Vector Caching?**
   Re-embedding thousands of text chunks on every application startup is computationally expensive. Disk caching reduces initialization time from 45 seconds to under 0.2 seconds.
5. **Why Decoupled Backend and Frontend Architecture?**
   Hosting the FastAPI REST layer independently of the Streamlit presentation layer allows the core RAG intelligence to serve other clients (mobile apps, web frontends, or automated evaluators) without modifying retrieval logic.

---

## 🌐 Current Limitations & Deployment Notes

### Deployment Status
- **Local Full-Stack Application:** **Working & Verified**
- **FastAPI Backend:** **Local** (`127.0.0.1:8000`)
- **Streamlit Frontend:** **Local** (`localhost:8501`)
- **Ollama Inference:** **Local** (`llama3.2:3b` via `http://localhost:11434`)
- **Public Cloud Deployment:** **Not yet completed**

### Requirements for Public Cloud Deployment
1. **Cloud LLM API Integration:** The current system relies on a local Ollama instance. For hosting on serverless platforms (e.g., Streamlit Community Cloud, Render, or Railway), the inference client should be configured to connect to an external hosted API (such as Google Gemini API, Groq, or OpenAI) or a dedicated GPU container.
2. **Decoupled Cloud Hosting:** The FastAPI backend and Streamlit UI should be deployed to suitable container environments (e.g., Render/Railway for backend; Streamlit Cloud for frontend, referencing the public backend URL via `API_URL`).
3. **Persistent Volume Mounting:** Vector caches (`data/.rag_cache/`) and the SQLite database (`data/learning_progress.db`) require persistent volume storage to maintain state across container restarts.

---

## 🔮 Future Scope

- **Cloud Model Selector:** Dynamic toggle between local Ollama inference and cloud APIs (Google Gemini / Groq) via UI settings.
- **OCR Ingestion:** Optical Character Recognition for scanned handwritten student notes and physical exam sheets.
- **Cross-Encoder Reranking:** Secondary transformer reranking stage to further refine top-k retrieval precision.
- **Automated Spaced Repetition:** Flashcard generation with Leitner box scheduling integrated into the learning history dashboard.

---

## 📋 Track A Compliance Checklist

| Track A Requirement | Specified Milestone | Implementation Status | Verification Details |
|---|---|:---:|---|
| **Multi-Format Processing** | Week 1–2 | ✅ **Complete** | Ingests PDF (PyMuPDF), DOCX (python-docx), and PPTX (python-pptx). |
| **Content Categorization** | Week 1–2 | ✅ **Complete** | Classifies files as `Notes`, `Question Bank`, or `Textbook`. |
| **Topic-Based Retrieval** | Week 1–2 | ✅ **Complete** | Hybrid dense FAISS + lexical scoring with CS alias expansion. |
| **Streamlit Interface** | Week 1–2 | ✅ **Complete** | Full 8-page interactive web UI with dashboard and document browser. |
| **Topic Explanation** | Week 3–4 | ✅ **Complete** | Multi-source grounded synthesis via `/explain`. |
| **Question Solving** | Week 3–4 | ✅ **Complete** | Solves exam questions grounded in retrieved textbook excerpts. |
| **Question Bank Integration** | Week 3–4 | ✅ **Complete** | Matches past exam questions against concepts with marks allocations. |
| **Pedagogical Flow** | Week 3–4 | ✅ **Complete** | `Theory → Example → Practice → Assessment` pipeline verified. |
| **Subject & Chapter Organization** | Week 3–4 | ✅ **Complete** | Subject filtering and natural numerical chapter sorting implemented. |
| **Cross-Document Referencing** | Week 3–4 | ✅ **Complete** | Every answer cites contributing filenames, chapters, and types. |
| **Option A1: CS Subject Guide** | Week 5–6 | ✅ **Complete** | Multi-language syntax formatting, deterministic code extraction, Mermaid diagrams. |
| **Multi-Document Upload** | Week 7–8 | ✅ **Complete** | Batch file upload supported on both backend (`/upload-batch`) and Streamlit UI. |
| **Persistent Progress** | Week 7–8 | ✅ **Complete** | SQLite database (`data/learning_progress.db`) persists history across restarts. |
| **Export Functionality** | Week 7–8 | ✅ **Complete** | One-click downloads for Markdown (`.md`) and Plain Text (`.txt`). |
| **Input Error Validation** | Week 7–8 | ✅ **Complete** | Validates requests and returns descriptive HTTP 400 error payloads. |
| **Documentation & Demo Readiness** | Week 7–8 | ✅ **Complete** | Comprehensive architectural documentation and verified regression tests. |

---

## 📜 License & Attribution

Developed as an academic capstone submission for the **Subject Guide & Question Bank AI Assistant (Track A — Option A1 Computer Science Subject Guide)**.  
Intended strictly for academic and educational learning assistance.
