# HRQuery System Architecture & Technical Specification

HRQuery is an AI-powered enterprise Retrieval-Augmented Generation (RAG) assistant designed for Human Resources and company policy guidance. It guarantees grounded, factual answers strictly extracted from verified company documentation with exact source attribution.

---

## 1. System Architecture Overview

```
========================================================================================
                                INGESTION PIPELINE (OFFLINE / SYNC)
========================================================================================
[ Company Documents ]  -->  [ DocumentLoader ]  -->  [ Paragraph Chunker ]
(PDF, DOCX, TXT)            • pypdf per-page         • Paragraph-aware
                            • python-docx flows      • 1000 char size
                            • UTF-8 decoder          • 150 char overlap
                                                             │
                                                             ▼
                                                    [ EmbeddingService ]
                                                    • text-embedding-004 (768-d)
                                                    • L2 Normalization (||v|| = 1.0)
                                                             │
                                                             ▼
                                                    [ VectorStoreManager ]
                                                    • FAISS IndexFlatIP (Cosine)
                                                    • Persistent metadata.json
                                                    • Content Hash Deduplication

========================================================================================
                                 QUERY PIPELINE (RUNTIME)
========================================================================================
[ Employee Question ]
         │
         ▼
[ Query Embedding ]  (Identical text-embedding-004 model & L2 Normalization)
         │
         ▼
[ FAISS Similarity Search ]  (Computes Cosine Similarity u · v across index)
         │
         ▼
[ Top-K Nearest Chunks ]  (Retrieves top-k candidates with similarity scores)
         │
         ├─── (Score < 0.38) ───► [ Relevance Fallback Gatekeeper ] ──► "I couldn't find enough information..."
         │
         ▼ (Score >= 0.38)
[ Structured Prompt Builder ]
  • System Prompt: Strict zero-hallucination guardrail
  • Injected Chunks with Document Name and Page Number
         │
         ▼
[ Google Gemini LLM ]  (gemini-2.5-flash / gemini-1.5-flash at temperature=0.0)
         │
         ▼
[ Grounded Answer + Source Attribution ]
  • Concise factual answer
  • Sources: Document Name — Page X (or Page unavailable)
```

---

## 2. Ingestion Pipeline Details

### 2.1 Multi-Format Document Processing
- **PDF (`pypdf`)**: Extracted page by page, preserving exact 1-indexed page numbers. Each page is mapped into a discrete document object containing `{document_name, page_number, source_type: "pdf", text}`.
- **DOCX (`python-docx`)**: Extracts full paragraph runs and structured table rows, mapping them into logical document blocks.
- **TXT**: Reads standard UTF-8 and ANSI/Latin-1 text with line-break normalization.
- **Deduplication**: Computes an MD5 checksum of the file bytes. If the identical document hash already exists in `metadata.json`, re-indexing is skipped to conserve API calls and avoid duplicate search vectors.

### 2.2 Paragraph-Aware Semantic Chunking
- **Target Size**: 1000 characters (~200–250 tokens).
- **Overlap**: 150 characters (~30–40 tokens).
- **Boundary Strategy**: Text is first partitioned along double newlines (`\n\n`) to preserve paragraph integrity. Over-sized paragraphs are subdivided at sentence boundaries (`[.!?]`). Sliding windows recombine units while ensuring no policy rule is separated from its eligibility clause or qualification condition.

### 2.3 Dense Vector Embeddings
- **Model**: Google GenAI `text-embedding-004` producing 768-dimensional dense float32 vectors.
- **L2 Normalization**: All chunk vectors are normalized to unit Euclidean length:
  $$\hat{v} = \frac{v}{\|v\|_2}$$
  This mathematical transformation guarantees that the Inner Product (dot product) between any two vectors equals their Cosine Similarity.

### 2.4 Vector Storage & Persistence
- **Index**: `faiss.IndexFlatIP(768)` (Inner Product search on normalized vectors).
- **Persistence**: Index is written to disk at `vector_store/index.faiss`. Chunk metadata (chunk ID, document name, page number, source type, and text) is stored alongside at `vector_store/metadata.json`.

---

## 3. Query & Retrieval Pipeline

### 3.1 Cosine Similarity Metric & Thresholding
Because all vectors are unit-normalized:
$$\text{Inner Product}(u, v) = u \cdot v = \cos(\theta) \in [-1.0, 1.0]$$
- **Score 1.0**: Exact semantic alignment.
- **Score >= 0.45**: High domain relevance (e.g. "casual leaves" matching `Leave_Policy.pdf` scores 0.53+).
- **Score < 0.38**: Irrelevant or out-of-scope (e.g. salary inquiries or cafeteria questions score ~0.32).

### 3.2 Two-Tier Grounding Strategy
1. **Tier 1 (Pre-LLM Gatekeeper)**: If the highest similarity score in the retrieved top-k candidate chunks is below `RELEVANCE_THRESHOLD` (0.38), the query is immediately rejected. The LLM is never invoked with noise, eliminating hallucination at the source and saving compute.
2. **Tier 2 (Prompt Constraint)**: The LLM is invoked with temperature 0.0 and a strict system prompt instructing it to respond only from context, and to output the exact required fallback phrase if the context does not explicitly provide the answer.

### 3.3 Exact Grounding Fallback Phrase
Whenever information is missing or out-of-scope, the system responds identically:
> *"I couldn't find enough information in the provided company documents to answer this question."*

### 3.4 Verified Source Attribution
Sources are extracted directly from the metadata of chunks used to answer the query:
```
Sources:
1. Leave_Policy.pdf — Page 1
2. Work_From_Home_Policy.pdf — Page 2
```
If the page number is unknown or not applicable (e.g. single-page TXT files), the system outputs `Page unavailable`. No source citations are ever fabricated.
