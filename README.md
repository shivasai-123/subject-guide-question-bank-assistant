# 📚 Subject Guide & Question Bank AI Assistant
### Multi-Document RAG Academic Learning Assistant
**Track A: Subject Study Guide & Academic Learning Assistant**  
**Selected Specialization: Option A1 — Computer Science Subject Guide**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-yellow.svg)](https://github.com/facebookresearch/faiss)
[![SentenceTransformers](https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-orange.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Ollama](https://img.shields.io/badge/Ollama-llama3.2%3A1b-purple.svg)](https://ollama.com/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-API%20Supported-4285F4.svg)](https://ai.google.dev/)
[![SQLite](https://img.shields.io/badge/SQLite-Persistent%20History-003B57.svg)](https://www.sqlite.org/)

---

## 🌟 Project Overview

The **Subject Guide & Question Bank AI Assistant** is a multi-document Retrieval-Augmented Generation (RAG) academic platform designed for university students, educators, and curriculum planners. University course materials are typically scattered across heterogeneous, unstructured files—including **Lecture Handouts**, **Textbooks**, **Lab Manuals**, and **Previous Year Question Papers (PYQs)**.

This platform bridges that gap by ingesting course documents, semantically indexing their contents, and synthesizing them into:
1. **Retrieval-Grounded Explanations:** Factual, syllabus-aligned conceptual explanations anchored directly in courseware without speculative hallucinations.
2. **Question Bank Integration:** Exam questions automatically mapped to textbook concepts with cited step-by-step solutions and marks-weightage context.
3. **Structured Learning Progression:** Step-by-step pedagogy adhering to the **Theory → Code / Example → Visual Diagram → Practice → Assessment** model.
4. **Computer Science (Option A1) Specialization:** Deterministic code extraction, syntax formatting (Python, C/C++, Java, SQL), and grounded Mermaid state flows for algorithms, database normalization, operating system lifecycles, and network protocol stacks.
5. **Exam Preparation Suite:** High-yield revision summaries, customizable multi-day revision planners (3 to 14 days), and diagnostic self-assessment quizzes with complete answer keys.
6. **One-Click Academic Exports:** Direct downloads of revision guides, quiz keys, and study notes in Markdown (`.md`) and Plain Text (`.txt`).
7. **Dual-Architecture LLM Support:** Flexible inference options supporting local, offline development via **Ollama** (`llama3.2:1b`) and cloud deployment via the **Google Gemini API** (`gemini-3.8-flash` with automatic fallback to `gemini-3.5-flash-lite`).

---

## ❗ Problem Statement

University academic resources are typically fragmented across heterogeneous file formats (PDF handouts, Word documents, PowerPoint lecture slides, and question papers). Consequently:
- **Disjointed Revision:** Students spend excessive study time manually cross-referencing past exam questions against hundreds of lecture slides to find answers.
- **Search Inefficiency:** Traditional keyword search ignores semantic intent, while pure vector search often suffers from semantic drift (e.g., conflating distinct graph algorithms like BFS and DFS due to high cosine similarity).
- **Code & Fact Hallucination:** Generic LLMs frequently synthesize non-standard code patterns, obsolete libraries, or unverified algorithm steps that diverge from university syllabi.
- **Session Volatility:** Standard learning prototypes retain history purely in volatile memory, discarding progress whenever the server restarts.

---

## 💡 Proposed Solution

This project implements a multi-format, dual-layer RAG system specifically engineered for academic curriculum mastery:
- **Unified Ingestion:** Extracts and standardizes text from PDF (PyMuPDF), DOCX (python-docx), and PPTX (python-pptx) files in single or batch uploads.
- **Content-Aware Indexing:** Automatically categorizes material as `Notes`, `Question Bank`, `Textbook`, or `Lab Manual`, using metadata weighting to prevent question prompts from polluting theoretical definitions.
- **Hybrid Retrieval:** Merges dense vector embeddings (`all-MiniLM-L6-v2`) with lexical keyword scoring and domain-specific query aliasing.
- **Deterministic Grounding:** Extracts genuine code blocks from course material and renders visual state progression diagrams (Mermaid) to visually reinforce algorithmic steps.
- **Persistent State:** Backed by an SQLite database for student progress tracking and a persistent disk cache for instant sub-second FAISS initialization.

---

## ✨ Key Capabilities

- **Multi-Format Document Ingestion:** Native parsing of `.pdf`, `.docx`, and `.pptx` documents with sliding-window chunking.
- **Batch / Multi-File Upload:** Ingest and index multiple documents simultaneously via the `/upload-batch` endpoint and the Streamlit UI.
- **Content Type Classification:** Differentiates study material from previous year question banks to tailor retrieval strategies.
- **Semantic Retrieval with Embeddings & FAISS:** Eliminates semantic drift using dense vector similarity combined with lexical keyword matching and technical query expansion.
- **Grounded AI Explanations:** Conceptual explanations anchored directly in retrieved textbook chunks with explicit source citations.
- **Question Solving & PYQ Mapping:** Locates exam questions matching any topic and synthesizes cited answers with marks allocation context.
- **Multi-Document Synthesis:** Combines insights across multiple lecture notes and textbooks for comparative study.
- **Pedagogical Progression (Option A1):** Generates structured learning paths: `Theory → Code/Example → Visual Diagram → Practice → Assessment`.
- **Grounded Algorithm Visualizations:** Generates Mermaid diagrams representing FIFO queue states in BFS, LIFO stack states in DFS, divide-and-conquer binary search, DBMS normalization levels, and OS process state transitions.
- **Quiz & Assessment Functionality:** Creates 5-question diagnostic assessments featuring MCQs, definition checks, code/application questions, and an answer key.
- **Exam Preparation & Multi-Day Planner:** Formulates customizable 3- to 14-day study plans and high-yield revision summaries.
- **Computer Science Code Support:** Multi-language syntax formatting for Python, C/C++, Java, and SQL with deterministic snippet extraction from uploaded notes.
- **Natural Chapter Sorting:** Dynamically sorts chapters numerically (`Chapter 1`, `Chapter 2`, `Chapter 10`) rather than alphabetically (`Chapter 1`, `Chapter 10`, `Chapter 2`).
- **Dashboard & Persistent History:** SQLite database (`learning_progress.db`) stores user activity across restarts; vector cache (`.rag_cache/`) eliminates re-embedding delays.
- **Academic Export:** One-click download of guides and quiz solutions formatted as `.md` or `.txt`.

---

## 🏛️ System Architecture

The application is engineered with a decoupled frontend, backend, retrieval, and persistent storage tier. It provides first-class support for both **local offline development** and **cloud production deployment**:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        Streamlit Web Interface                         │
│                    (frontend/streamlit_app.py)                         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP REST API (API_URL)
┌───────────────────────────────────▼────────────────────────────────────┐
│                         FastAPI Backend                                │
│                       (backend/app.py)                                 │
└─────┬─────────────────────┬──────────────────────┬───────────────┬─────┘
      │                     │                      │               │
┌─────▼───────────────┐ ┌───▼────────────────┐ ┌───▼───────────┐ ┌──▼───────────┐
│ Document Processing │ │  Metadata Manager  │ │ Hybrid RAG    │ │ LRU Response  │
│ (PyMuPDF/docx/pptx) │ │ (document_metadata)│ │ (FAISS Dense  │ │ Query Cache   │
└─────────────────────┘ └────────────────────┘ │ + Lexical)    │ └───────────────┘
                                               └───────┬───────┘
                                                       │
        ┌──────────────────────────────────────────────┴──────────────────────────────────────────────┐
        │                                                                                             │
        ▼ (Local Development)                                                                         ▼ (Cloud Deployment)
┌──────────────────────────────────────────────┐                              ┌──────────────────────────────────────────────┐
│                LOCAL INFERENCE               │                              │               CLOUD INFERENCE                │
│             LLM_PROVIDER="ollama"            │                              │             LLM_PROVIDER="gemini"            │
│                                              │                              │                                              │
│  • Ollama Server (http://localhost:11434)    │                              │  • Google Gemini API (google-genai SDK)      │
│  • Default Model: llama3.2:1b                │                              │  • Primary Model: gemini-3.8-flash           │
│  • Zero API costs, 100% private & offline    │                              │  • Resilient Fallback: gemini-3.5-flash-lite │
│  • Tuned for 8 GB RAM local machines         │                              │  • Automatic 1x retry on 503 / UNAVAILABLE   │
└──────────────────────────────────────────────┘                              └──────────────────────────────────────────────┘
```

### Why Dual Architectures?

1. **Local Architecture (`Streamlit → FastAPI → Ollama → llama3.2:1b`):**
   - **Privacy & Offline Access:** Ideal for local development, classroom environments, or evaluating sensitive institutional materials without transmitting data to external third parties.
   - **Zero API Costs:** Runs completely free on local hardware.
   - **Optimized Footprint:** Uses `llama3.2:1b` with bounded context (`2500` characters) and generation caps (`280` tokens), ensuring responsive responses on standard 8 GB RAM laptops.

2. **Cloud Architecture (`Streamlit Community Cloud → Render FastAPI backend → Google Gemini API`):**
   - **Deployment Separation:** Ollama is intended for local development and private testing, while the Gemini API is used for cloud deployment so the hosted Render backend does not need to run a local Ollama model.
   - **Scalable Cloud Inference:** Cloud inference uses Google's Gemini API to provide scalable model inference without requiring local LLM hardware.
   - **High Availability:** Includes an automatic fallback to `gemini-3.5-flash-lite` if the primary model encounters temporary capacity limitations or HTTP 503 errors.

---

## 🔍 The RAG Pipeline

In simple terms, the Retrieval-Augmented Generation pipeline operates in sequential stages:

$$\text{Document} \longrightarrow \text{Text Extraction} \longrightarrow \text{Chunking} \longrightarrow \text{Embeddings} \longrightarrow \text{FAISS Vector Search} \longrightarrow \text{Relevant Chunks} \longrightarrow \text{LLM} \longrightarrow \text{Grounded Answer}$$

### Detailed Processing Stages

```text
User Question / Topic
       │
       ▼
[1] Query Intent Classifier & Alias Expansion
       ├── Identifies intent (conceptual, question-bank, exam-prep)
       └── Expands technical terms (e.g., "BFS" -> "breadth first search", "queue", "level order")
       │
       ▼
[2] Hybrid Candidate Retrieval
       ├── Dense Vector Search: FAISS IndexFlatL2 over 384-d MiniLM embeddings
       └── Lexical Keyword Scoring: Exact match boost on technical terms
       │
       ▼
[3] Precision Filtering & Metadata Gate
       ├── Discards chunks below similarity threshold
       └── Filters by Subject, Chapter, and Content Type (Notes vs. Question Bank)
       │
       ▼
[4] Context Assembly & Source Attribution
       └── Formats top-k chunks with filename, chapter, and page citations
       │
       ▼
[5] Guardrailed LLM Synthesis (Ollama or Gemini)
       ├── Injects syllabus-bounded system prompts
       └── Deterministically extracts code blocks and renders Mermaid diagrams
       │
       ▼
Verified Answer with Source Citation Cards
```

---

## 🤖 Supported LLM Providers

The application dynamically selects the LLM provider based on the `LLM_PROVIDER` environment variable:

| Feature | Local Provider (`ollama`) | Cloud Provider (`gemini`) |
|---|---|---|
| **Configuration** | `LLM_PROVIDER=ollama` (Default) | `LLM_PROVIDER=gemini` |
| **Model** | `llama3.2:1b` (via `OLLAMA_MODEL`) | `gemini-3.8-flash` (via `GEMINI_MODEL`) |
| **Fallback Model** | N/A | `gemini-3.5-flash-lite` (via `GEMINI_FALLBACK_MODEL`) |
| **SDK / Driver** | `ollama-python` | `google-genai` |
| **API Key Required** | ❌ No | ✅ Yes (`GEMINI_API_KEY`) |
| **Internet Access** | ❌ Offline | ✅ Required |
| **Fallback Behavior** | Standard error message on failure | Automatic 1-time retry on temporary 503 / `UNAVAILABLE` errors |
| **Primary Use Case** | Local development, testing, privacy | Cloud deployment (Render, Streamlit Cloud) |

### Resilient Gemini Fallback Logic
When `LLM_PROVIDER=gemini`:
1. The assistant initiates generation using `GEMINI_MODEL` (`gemini-3.8-flash`).
2. If Google's API returns a temporary availability or capacity error (specifically HTTP 503 or `UNAVAILABLE`), the system intercepts the error, logs a performance warning, and automatically retries once using `GEMINI_FALLBACK_MODEL` (`gemini-3.5-flash-lite`).
3. Client errors (such as HTTP 400 Invalid Argument or HTTP 401 Bad API Key) do **not** trigger the fallback.
4. If the fallback also fails, the system safely returns the standard user-facing error message without crashing the server.

---

## 🛠️ Technology Stack

Every technology listed below is actively used in the codebase:

| Component | Technology | Version | Purpose in Codebase |
|---|---|---|---|
| **Language** | Python | `>= 3.10` | Core application programming language |
| **REST Backend** | FastAPI | `0.141.1` | High-performance asynchronous REST API routing and endpoints |
| **ASGI Server** | Uvicorn | `0.52.4` | Production ASGI web server |
| **Frontend UI** | Streamlit | `1.64.0` | Multi-page interactive web dashboard |
| **Vector Search** | FAISS (`faiss-cpu`) | `1.15.0` | Dense L2 similarity vector indexing and nearest-neighbor search |
| **Embeddings** | Sentence-Transformers | `6.0.0` | `all-MiniLM-L6-v2` dense 384-dimensional text representations |
| **Local LLM** | Ollama | `llama3.2:1b` | Local prompt-constrained inference engine |
| **Cloud LLM** | Google Gemini API (`google-genai`) | `2.28.0` | Cloud inference with automated availability fallback |
| **Database** | SQLite 3 | Built-in | Relational persistent learning progress and audit logging |
| **PDF Extraction** | PyMuPDF (`fitz`) | `1.28.2` | High-speed PDF text parsing and paragraph reconstruction |
| **Word Extraction**| python-docx | `1.2.0` | Parsing DOCX paragraphs and structural tables |
| **Slide Extraction**| python-pptx | `1.0.2` | Extracting PowerPoint text frames and slide notes |
| **Machine Learning**| scikit-learn | `1.9.0` | Vector similarity calculations and mathematical utilities |
| **Data Validation**| Pydantic | Built-in | Request/response schema models and input validation |
| **HTTP Client** | Requests | `2.34.2` | REST API communication between Streamlit and FastAPI |

---

## 📂 Project Structure

```text
subject-guide-question-bank-assistant/
│
├── backend/
│   ├── app.py                      # FastAPI REST application & endpoints
│   ├── document_processor.py       # Multi-format document parser (PDF, DOCX, PPTX)
│   ├── learning_tools.py           # Multi-provider LLM prompting, diagrams & SQLite
│   ├── metadata_manager.py         # Metadata schema & document catalog
│   ├── rag_engine.py               # Hybrid FAISS vector store & query routing
│   └── requirements.txt            # Backend Python dependencies
│
├── frontend/
│   ├── .streamlit/
│   │   └── config.toml             # Streamlit client toolbar configuration
│   ├── requirements.txt            # Frontend Python dependencies (Streamlit, requests)
│   └── streamlit_app.py            # Streamlit interactive web application (port 8501)
│
├── data/
│   ├── document_metadata.json      # Tracked catalog metadata (subject, chapter, type)
│   ├── learning_progress.db        # Persistent SQLite learning history (git-ignored)
│   └── .rag_cache/                 # Persisted FAISS index & embeddings (git-ignored)
│
├── static/
│   └── index.html                  # Standalone single-page web interface (served at /)
│
├── tests/
│   ├── app_backup.py               # Baseline archive application
│   ├── embedding_test.py           # Embedding sanity test script
│   ├── ollama_test.py              # Ollama connectivity test script
│   ├── project.py                  # Prototype test script
│   └── similarity_test.py          # Vector cosine similarity test script
│
├── .env.example                    # Sample environment variables template
├── .gitignore                      # Git exclusion rules (venv, caches, private docs)
├── .streamlit/
│   └── config.toml                 # Root Streamlit client configuration
└── README.md                       # Comprehensive project documentation
```

---

## 🔒 Data & Privacy

> [!IMPORTANT]
> The public repository intentionally **does NOT include private study materials**.

The following files are strictly excluded via `.gitignore`:
- `data/*.pdf`
- `data/*.pptx`
- `data/*.docx`
- `data/.rag_cache/`
- `data/learning_progress.db`
- `*.log`, `*.db`, `*.sqlite3`
- `.env`, `.env.local`

**Why?** University courseware, lecture slides, and copyrighted textbook excerpts often contain institutional or copyright-restricted material. Users should place their own syllabus documents in the `data/` folder locally or upload them directly via the Streamlit web dashboard. Do **not** commit private academic documents to public repositories.

---

## ⚙️ Environment Variables

The project uses environment variables for clean configuration management. A template is provided in [`.env.example`](file:///.env.example):

| Variable | Default Value | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | Active LLM backend (`ollama` or `gemini`). |
| `OLLAMA_MODEL` | `llama3.2:1b` | Local Ollama model tag. |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Primary Gemini model for cloud inference. |
| `GEMINI_FALLBACK_MODEL` | `gemini-3.5-flash-lite` | Fallback model used on temporary 503 capacity errors. |
| `GEMINI_API_KEY` | *(empty)* | Google Gemini API key (required when `LLM_PROVIDER=gemini`). |
| `API_URL` | `http://127.0.0.1:8000` | Backend URL referenced by the Streamlit frontend. |
| `DATA_FOLDER` | `data` | Path to document storage directory. |
| `CACHE_FOLDER` | `data/.rag_cache` | Path to FAISS vector index cache directory. |
| `DB_PATH` | `data/learning_progress.db` | Path to SQLite learning history database. |

---

## 🚀 Local Setup & Installation

### 1. Prerequisites
- **Python:** `3.10`, `3.11`, or `3.12` installed.
- **Ollama:** Installed from [ollama.com](https://ollama.com/) (if using the local provider).

### 2. Setup Steps

```powershell
# 1. Clone the repository
git clone https://github.com/shivasai-123/subject-guide-question-bank-assistant.git
cd subject-guide-question-bank-assistant

# 2. Create Python virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

# 4. Install backend dependencies
pip install -r backend/requirements.txt

# 5. Install frontend dependencies
pip install -r frontend/requirements.txt

# 6. Configure environment variables
# Copy the example file to .env
cp .env.example .env
```

### 3. Starting Ollama (Local Provider)
If you are using the default local `ollama` provider:
```powershell
# Pull the lightweight 1B model
ollama pull llama3.2:1b

# Ensure the Ollama daemon is running
ollama serve
```

---

## 🏃 Running the Application

Launch the backend and frontend in separate terminal windows:

### Terminal 1: FastAPI Backend (Port 8000)
From the project root:
```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```
- Interactive Swagger API Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Built-in Static Web Client: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

### Terminal 2: Streamlit Interactive UI (Port 8501)
From the project root:
```powershell
.\.venv\Scripts\streamlit.exe run frontend/streamlit_app.py
```
- Streamlit Web Dashboard opens automatically at: [http://localhost:8501](http://localhost:8501)

---

## 🔑 Google Gemini API Configuration

To switch the application from local Ollama to the Google Gemini cloud API:

1. Obtain an API key from [Google AI Studio](https://aistudio.google.com/).
2. In your `.env` file (or terminal session), configure:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.8-flash
GEMINI_FALLBACK_MODEL=gemini-3.5-flash-lite
```

> [!WARNING]
> Never hard-code or commit your actual `GEMINI_API_KEY` into `README.md` or git history. Always manage keys via `.env` or cloud secret managers.

---

## ☁️ Deployment Architecture (Planned / Ready)

The application is structured to be deployed cleanly onto cloud infrastructure:

### Backend Hosting: Render
The FastAPI service can be hosted on [Render](https://render.com/) as a Python Web Service:
- **Build Command:**
  ```bash
  pip install -r backend/requirements.txt
  ```
- **Start Command:**
  ```bash
  uvicorn backend.app:app --host 0.0.0.0 --port $PORT
  ```
- **Environment Variables on Render:**
  ```env
  LLM_PROVIDER=gemini
  GEMINI_API_KEY=your_api_key_here
  GEMINI_MODEL=gemini-3.8-flash
  GEMINI_FALLBACK_MODEL=gemini-3.5-flash-lite
  PYTHON_VERSION=3.11.0
  ```

### Frontend Hosting: Streamlit Community Cloud
The Streamlit interface can be hosted directly on [Streamlit Community Cloud](https://streamlit.io/cloud):
- **Repository:** `your-username/subject-guide-question-bank-assistant`
- **Main file path:** `frontend/streamlit_app.py`
- **App Secret / Environment Variable:**
  ```toml
  API_URL = "https://your-backend-app.onrender.com"
  ```

*Note: The repository is deployment-ready. Public deployment URLs can be added here once hosted on your cloud account.*

---

## ⚡ Performance & Optimizations

All performance enhancements reflect actual mechanisms implemented in the codebase:

1. **Persistent Vector Disk Caching (`data/.rag_cache/`):**
   Stores serialized FAISS binary indices and NumPy embedding matrices. Sub-second engine initialization (`0.03s - 0.17s`) avoids re-embedding thousands of chunks on server restart.
2. **In-Memory LRU Response Cache (`ResponseCache`):**
   `backend/app.py` implements an in-memory LRU query cache (capacity: 128) that immediately returns cached answers for identical queries without invoking the LLM.
3. **Optimized 1B Model Parameter Footprint:**
   Transitioning the default local model to `llama3.2:1b` reduced prompt evaluation and generation latency significantly compared to heavier 3B+ models on 8 GB RAM machines.
4. **Targeted Context & Generation Limits:**
   Input context is capped at `2500` characters and generation output is bounded at `280` tokens to prevent runaway loops and ensure concise academic answers.
5. **Tuned Ollama Runtime Options:**
   Inference requests specify `num_ctx: 1536`, `num_thread: 4`, and `keep_alive: "15m"`, preventing unnecessary model unloading and thread contention.
6. **Streamlit Client-Side Data Caching:**
   `frontend/streamlit_app.py` uses `@st.cache_data(ttl=15)` for `/subjects` and `/chapters` endpoints to prevent redundant network round-trips.

---

## 🧪 Testing & Verification

### Backend Import & Sanity Verification
Verify that all backend modules, FAISS, and PyTorch embeddings load cleanly:

```powershell
.\.venv\Scripts\python.exe -c "import backend.app; print('BACKEND IMPORT OK')"
```

**Expected output:**
```text
==========================================
INITIALIZING RAG ENGINE
==========================================
Loading embedding model: sentence-transformers/all-MiniLM-L6-v2
Embedding dimension: 384
RAG engine ready.
==========================================
Loaded RAG cache: ... vectors.
BACKEND IMPORT OK
```

### Health Check Verification
With the backend running, query the health endpoint:
```powershell
curl http://127.0.0.1:8000/health
```
**Sample response:**
```json
{
  "status": "running",
  "documents": 7,
  "vectors": 2156,
  "subjects": ["C-Programming", "DM", "General", "Python"],
  "model": "llama3.2:1b",
  "supported_formats": ["PDF", "DOCX", "PPTX"]
}
```

---

## 📋 Deployment Checklist

Before publishing or deploying to production, verify each item:

- [ ] **Dependencies Updated:** `backend/requirements.txt` and `frontend/requirements.txt` reflect exact requirements.
- [ ] **Environment Variables Configured:** `.env` exists locally; cloud secrets set on Render/Streamlit Cloud.
- [ ] **Private Documents Excluded:** `.gitignore` excludes `data/*.pdf`, `data/*.docx`, `data/*.pptx`, and `.rag_cache/`.
- [ ] **No Secrets in Repo:** No actual API keys or passwords committed.
- [ ] **Backend Tested Locally:** `import backend.app` succeeds with `BACKEND IMPORT OK`.
- [ ] **Frontend Tested Locally:** Streamlit dashboard loads cleanly with all 8 pages functional.
- [ ] **Render Backend Deployed:** Web service created with start command `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`.
- [ ] **Streamlit Cloud Deployed:** App points to `frontend/streamlit_app.py` with `API_URL` set to the Render backend URL.
- [ ] **Public Health Check Tested:** Render URL `/health` returns status `200`.
- [ ] **Public Web App Tested:** End-to-end question answering and topic learning verified.

---

## 🔌 API Overview

The FastAPI backend (`backend/app.py`) provides the following REST endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the static single-page application (`static/index.html`). |
| `GET` | `/health` | Returns server health, document/vector counts, model info, and formats. |
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

## 🖥️ Streamlit Interface Views

The frontend interface (`frontend/streamlit_app.py`) provides 8 dedicated modules:

1. **🏠 Dashboard:** Overview metrics (documents indexed, vector count, subjects, content distribution) and recent uploads.
2. **💬 Ask Question:** Retrieval-grounded question answering with subject/chapter filtering and source citation cards.
3. **🧠 Learn a Topic:** 4-stage pedagogical exploration (Theory, Code Example, Visual Diagram, Practice, Assessment).
4. **📝 Question Bank:** Search previous year questions by topic or browse entire chapter collections.
5. **🎯 Exam Prep & Planner:** High-yield exam guides, multi-day revision schedules, and diagnostic self-quizzes.
6. **📄 Documents:** Document library browser displaying subjects, chapters, chunks, and download buttons.
7. **⬆️ Upload Document:** Drag-and-drop batch uploader supporting multiple PDFs, DOCXs, and PPTXs with progress bars.
8. **📜 Learning History:** SQLite-backed audit log of user learning sessions with search and history-clear options.

---

## 💻 Option A1 — Computer Science Subject Guide

As the selected Track A specialization, Option A1 provides specialized processing for core Computer Science domains:

### 1. Multi-Language Code Handling
- **Language Detection & Classification:** Automatically recognizes and syntax-highlights **Python**, **C/C++**, **Java**, and **SQL** snippets.
- **Deterministic Extraction:** Isolates genuine code directly from notes chunks to prevent LLM hallucinations.

### 2. Grounded Algorithmic State Diagrams (Mermaid)
For algorithm and systems topics, the engine injects structured Mermaid diagrams:
- **Breadth-First Search (BFS):** Graph topology and sequential FIFO queue state trace.
- **Depth-First Search (DFS):** Graph topology and sequential LIFO recursion stack trace.
- **Binary Search:** Array midpoint division trace (`Low`, `Mid`, `High`).
- **Database Normalization (DBMS):** Relational dependency hierarchy (`1NF` → `2NF` → `3NF` → `BCNF`).
- **Operating Systems (OS):** 5-state process lifecycle (`New` → `Ready` → `Running` → `Waiting` → `Terminated`) and deadlock models.
- **Computer Networks:** OSI 7-Layer protocol hierarchy and TCP/IP stack mappings.

---

## 📜 License & Attribution

Developed as an academic capstone submission for the **Subject Guide & Question Bank AI Assistant (Track A — Option A1 Computer Science Subject Guide)**.  
Intended strictly for academic and educational learning assistance.
