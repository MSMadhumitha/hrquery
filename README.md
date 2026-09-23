# HRQuery — AI-Powered HR & Company Policy Knowledge Assistant

> **A production-ready Retrieval-Augmented Generation (RAG) assistant for employee HR and corporate policy queries, enforcing strict context grounding, dual-tier hallucination guardrails, and verified source citations.**

---

## 📌 Problem Statement

In modern organizations, HR guidelines, company policies, travel rules, and employee handbooks are distributed across dozens of multi-page PDFs, Word documents, and text files. When employees have urgent policy questions (e.g., parental leave entitlements, remote work permissions during medical issues, or expense submission deadlines):
1. **Time Waste**: Employees spend hours sifting through dense manuals or waiting for HR email responses.
2. **Hallucination Risk**: Generic AI chatbots often guess or hallucinate policy rules, producing inaccurate commitments.
3. **Lack of Attribution**: Traditional chatbots rarely point to the exact document and page number where a rule is documented.

---

## 💡 Solution

**HRQuery** solves this challenge with a true end-to-end RAG architecture:
- Ingests multiple policy documents across **PDF, DOCX, and TXT** formats.
- Performs **paragraph-aware semantic chunking** that preserves logical policy clauses.
- Embeds content using Google's **`text-embedding-004`** model with L2 normalization.
- Persists vectors and metadata in a **FAISS** vector store with duplicate content detection.
- Employs **Cosine Similarity search** with a calibrated **relevance threshold** that gates out-of-scope inquiries before LLM invocation.
- Generates answers using Google **Gemini** (`gemini-2.5-flash` / `gemini-1.5-flash`) at temperature 0.0 with strict grounding instructions.
- Delivers exact source citations (`Document Name — Page X`) and triggers an exact fallback message when information is absent.

---

## 🌟 Key Features

- **Multi-Format Ingestion**: Process PDFs (page-by-page), DOCX documents, and TXT files simultaneously.
- **True RAG Pipeline**: No prompt stuffing. Documents are chunked, embedded, indexed, and retrieved on-the-fly.
- **Dual-Tier Grounding**:
  1. *Retrieval Gatekeeper*: Blocks irrelevant/out-of-scope queries (e.g. salary inquiries or lunch menus) if similarity score < 0.38.
  2. *LLM Prompt Guardrail*: Zero-temperature prompt instructing Gemini to answer solely from context.
- **Strict Grounding Fallback**: Outputs the exact required fallback phrase whenever context is insufficient:
  > *"I couldn't find enough information in the provided company documents to answer this question."*
- **Source Attribution**: Always shows source document name and page number (or `"Page unavailable"`). Never fabricates citations.
- **Interactive Streamlit UI**: Complete with file drag-and-drop, knowledge base metrics, quick demo query chips, and a real-time **Debug Inspector** showing retrieved chunks and similarity scores.
- **Offline / Test Resilient**: Operates seamlessly with live Google Gemini API keys or with deterministic offline test vectorization for headless automated CI/CD.

---

## 🏗️ Architecture Diagram

```
Documents (PDF, DOCX, TXT)
          │
          ▼
   DocumentLoader (pypdf, python-docx, text decoder)
          │
          ▼
   Paragraph Chunker (1000 char target, 150 char overlap)
          │
          ▼
   EmbeddingService (text-embedding-004, 768-d, L2 Normalized)
          │
          ▼
   VectorStoreManager (FAISS IndexFlatIP + metadata.json)
          │
══════════════════════════════════════════════════════════════════
                      QUERY TIME PIPELINE
══════════════════════════════════════════════════════════════════
   User Question ──► Query Embedding (Normalized) ──► FAISS Search (Top-K)
                                                            │
                     ┌──────────────────────────────────────┘
                     ▼
          [ Relevance Gatekeeper ]
                     │
       ┌─────────────┴─────────────┐
       ▼ (Score < 0.38)            ▼ (Score >= 0.38)
Exact Fallback Message       Prompt + Top Chunks
("I couldn't find...")             │
                                   ▼
                             Google Gemini LLM (temp=0.0)
                                   │
                                   ▼
                             Grounded Answer + Sources
```

A visual architecture diagram is available in [`docs/architecture.png`](docs/architecture.png) and detailed specifications in [`docs/architecture.md`](docs/architecture.md).

---

## 🛠️ Tech Stack

| Component | Technology | Description |
|---|---|---|
| **Language** | Python 3.11+ / 3.14 | Core backend runtime |
| **Frontend UI** | Streamlit | Responsive web application |
| **LLM** | Google Gemini (`google-genai` SDK) | `gemini-2.5-flash` / `gemini-1.5-flash` |
| **Embeddings** | Google GenAI `text-embedding-004` | 768-dimensional semantic embeddings |
| **Vector DB** | FAISS (`faiss-cpu`) | In-memory & disk-persisted `IndexFlatIP` |
| **Document Processing**| `pypdf`, `python-docx` | Page-aware PDF and structured DOCX parsing |
| **Config & Secrets** | `python-dotenv` | Environment variable management |
| **Testing** | `pytest` | Automated unit and integration testing suite |

---

## 📁 Project Structure

```
hrquery-rag/
├── app.py                      # Streamlit interactive web application
├── pipeline.py                 # End-to-end RAG orchestrator
├── config.py                   # Centralized configuration & environment loader
├── requirements.txt            # Minimal, pinned dependencies
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore rules (protects API keys & cache)
├── README.md                   # Complete documentation & usage guide
├── data/
│   ├── sample_documents/       # 8 fictional NovaTech Solutions policies (PDF, DOCX, TXT)
│   └── uploads/                # Directory for user-uploaded documents
├── vector_store/               # Persistent FAISS index (index.faiss) & metadata.json
├── ingestion/
│   ├── __init__.py
│   ├── document_loader.py      # PDF, DOCX, and TXT loaders with page tracking
│   ├── chunker.py              # Paragraph-aware semantic text chunker
│   └── embedding.py            # Gemini text-embedding-004 & offline test fallback
├── retrieval/
│   ├── __init__.py
│   ├── vector_store.py         # FAISS IndexFlatIP manager & content-hash deduplication
│   └── retriever.py            # Top-K retrieval, Cosine scoring & thresholding
├── generation/
│   ├── __init__.py
│   ├── prompt.py               # Grounded system prompt & context formatting
│   └── llm.py                  # Gemini LLM interface & relevance gatekeeper
├── utils/
│   ├── __init__.py
│   ├── helpers.py              # Text cleaning, MD5 hashing, source formatting
│   └── generate_sample_docs.py # Automated generator for 8 realistic sample policies
├── docs/
│   ├── architecture.md         # Full technical specification & metric math
│   ├── architecture.png        # Architecture diagram graphic
│   └── generate_architecture_diagram.py
└── tests/
    ├── __init__.py
    ├── test_chunking.py        # Chunk size, overlap, paragraph boundaries, metadata
    ├── test_retrieval.py       # Loaders, vector persistence, query routing
    └── test_grounding.py       # Relevance thresholding, exact fallback, citations
```

---

## 🚀 Setup & Installation

### 1. Clone & Create Virtual Environment
```bash
git clone <repo-url>
cd HR-Rag

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (Command Prompt / PowerShell):
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Key
Create a `.env` file in the project root:
```bash
copy .env.example .env
```
Edit `.env` to include your Google Gemini API key:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```
*(Note: You can also enter the API key directly in the Streamlit UI sidebar.)*

### 4. Generate Sample Policy Documents (Included)
The 8 fictional NovaTech Solutions policy documents are already pre-generated in `data/sample_documents/`. To regenerate them at any time:
```bash
python utils/generate_sample_docs.py
```

### 5. Launch the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🎯 Demo Questions & Verification

The app includes interactive one-click buttons for all standard hackathon test queries:

| # | Demo Question | Target Policy Document | Expected Behavior / Answer |
|---|---|---|---|
| **1** | *"How many casual leaves are employees entitled to?"* | `Leave_Policy.pdf` | **Grounded Answer**: Entitled to 12 days/year (3/quarter); 48 hours notice. Citations: `Leave_Policy.pdf — Page 1`. |
| **2** | *"Can an employee work from home during a temporary health issue?"* | `Work_From_Home_Policy.pdf` | **Grounded Answer**: Yes; eligible for up to 4 consecutive weeks of remote work with doctor's note and manager approval. |
| **3** | *"What happens if an employee violates the company's code of conduct?"* | `Code_of_Conduct.pdf` | **Grounded Answer**: Investigated within 10 days; progressive discipline (written warning -> 14-day unpaid suspension -> termination). |
| **4** | *"How are travel expenses reimbursed?"* | `Travel_and_Expense_Policy.pdf` | **Grounded Answer**: Submit via NovaExpense within 15 days; itemized receipts for >$25; per diem $75 domestic / $120 intl. |
| **5** | *"What is the resignation notice period?"* | `Resignation_Policy.txt` | **Grounded Answer**: 30 days for Individual Contributors; 60 days for Managers; 90 days for Directors/Executives. |
| **6** | *"What is the salary of a software engineer with three years of experience?"* | *None (Out-of-Scope)* | **Grounding Fallback Triggered**: *"I couldn't find enough information in the provided company documents to answer this question."* |
| **7** | *"What is the cafeteria menu?"* | *None (Out-of-Scope)* | **Grounding Fallback Triggered**: *"I couldn't find enough information in the provided company documents to answer this question."* |

---

## 🔍 Grounding & Retrieval Strategy

### Metric Explanation & Threshold Justification
1. **L2 Normalization**: Every document chunk embedding and query embedding is normalized such that $\|\vec{v}\|_2 = 1.0$.
2. **Metric**: We employ FAISS `IndexFlatIP` (Inner Product). For unit-normalized vectors:
   $$\text{Inner Product} = \vec{u} \cdot \vec{v} = \cos(\theta)$$
   This directly yields exact **Cosine Similarity** bounded in the range $[-1.0, 1.0]$.
3. **Threshold Calibration**:
   - In-scope queries matching specific HR clauses consistently achieve similarity scores between **0.45 and 0.58**.
   - Irrelevant questions (e.g. software engineer compensation or cafeteria meals) score between **0.25 and 0.34**.
   - Setting `RELEVANCE_THRESHOLD = 0.38` establishes an optimal decision boundary that cleanly admits authentic policy questions while strictly intercepting out-of-scope inquiries.

### Why Chunking Matters for Retrieval Quality
- **Context Dilution Prevention**: Ingesting whole documents degrades vector resolution because an entire PDF is compressed into one vector, drowning out specific rules. Chunking into 1000-character segments maintains crisp semantic focus.
- **Paragraph Preservation**: Splitting arbitrarily across token counts fractures clauses. Our paragraph-aware chunker preserves paragraph headers and complete provisions together.
- **Sliding Overlap**: A 150-character overlap ensures rules spanning chunk seams are never truncated.

---

## 🧪 Automated Testing

HRQuery includes a comprehensive automated test suite testing chunking, retrieval accuracy, vector persistence, relevance thresholding, and source attribution:

Run all tests:
```bash
python -m pytest tests/ -v
```

Expected output:
```
tests/test_chunking.py::test_chunker_initialization PASSED               [ 10%]
tests/test_chunking.py::test_chunker_preserves_metadata PASSED           [ 20%]
tests/test_chunking.py::test_chunker_paragraph_boundary_awareness PASSED [ 30%]
tests/test_chunking.py::test_chunker_empty_input PASSED                  [ 40%]
tests/test_grounding.py::test_grounding_fallback_exact_string PASSED     [ 50%]
tests/test_grounding.py::test_out_of_scope_questions_trigger_fallback PASSED [ 60%]
tests/test_grounding.py::test_source_attribution_formatting PASSED       [ 70%]
tests/test_retrieval.py::test_document_loader_formats PASSED             [ 80%]
tests/test_retrieval.py::test_vector_store_persistence PASSED            [ 90%]
tests/test_retrieval.py::test_retrieval_accuracy_on_hr_queries PASSED    [100%]

======================== 10 passed in 4.30s ========================
```

---

## ⚠️ Limitations & Future Improvements

### Current Limitations:
1. **Scanned PDF OCR**: PDFs without native embedded text layers (e.g., pure image scans) require OCR preprocessing before text extraction.
2. **Complex Embedded Tables**: Multi-column nested tables in DOCX are serialized as markdown rows, which works well for policy clauses but may need specialized parsing for complex mathematical matrices.

### Future Improvements:
1. **Hybrid Search (BM25 + Dense FAISS)**: Combining dense semantic embeddings with BM25 keyword matching using Reciprocal Rank Fusion (RRF) for enhanced acronym and policy code retrieval.
2. **Cross-Encoder Reranking**: Introducing a lightweight cross-encoder (e.g. `bge-reranker`) on top-20 candidates before passing the top-5 to Gemini.
3. **Role-Based Access Control (RBAC)**: Tagging chunks with employee clearance levels (e.g. General Staff vs. Executive HR).

---

## 🏁 Starting the Application

To run the application right now:
```bash
streamlit run app.py
```
