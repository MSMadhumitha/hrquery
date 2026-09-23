"""
End-to-End RAG Pipeline Orchestrator for HRQuery.
Connects DocumentLoader, Chunker, EmbeddingService, VectorStoreManager,
Retriever, and GeminiRAGGenerator.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Union, BinaryIO, Optional

from config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    DEFAULT_RELEVANCE_THRESHOLD,
    SAMPLE_DOCS_DIR,
    FAISS_INDEX_PATH,
    METADATA_STORE_PATH
)
from ingestion.document_loader import DocumentLoader
from ingestion.chunker import TextChunker
from ingestion.embedding import EmbeddingService
from retrieval.vector_store import VectorStoreManager
from retrieval.retriever import HRQueryRetriever
from generation.llm import GeminiRAGGenerator
from utils.helpers import compute_file_hash

logger = logging.getLogger(__name__)


class HRQueryPipeline:
    """
    Main orchestrator for document ingestion, indexing, retrieval, and grounded response generation.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        top_k: int = DEFAULT_TOP_K,
        relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD
    ):
        self.api_key = api_key
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_service = EmbeddingService(api_key=api_key)
        self.vector_store = VectorStoreManager()
        self.retriever = HRQueryRetriever(
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
            top_k=top_k,
            relevance_threshold=relevance_threshold
        )
        self.generator = GeminiRAGGenerator(api_key=api_key)

    def set_api_key(self, api_key: str, model: Optional[str] = None) -> None:
        """Dynamically updates API key across services."""
        self.api_key = api_key
        self.embedding_service = EmbeddingService(api_key=api_key)
        self.retriever.embedding_service = self.embedding_service
        self.generator = GeminiRAGGenerator(api_key=api_key, model=model or self.generator.model)

    def set_model(self, model: str) -> None:
        """Dynamically updates the generative model."""
        self.generator = GeminiRAGGenerator(api_key=self.api_key, model=model)


    def ingest_document(
        self,
        source: Union[str, Path, BinaryIO],
        filename: str = ""
    ) -> Dict[str, Any]:
        """
        Ingests a single document:
        extract pages -> hash check -> chunk -> embed -> store in FAISS.
        """
        # Determine filename and compute hash
        if isinstance(source, (str, Path)):
            path = Path(source)
            fname = filename or path.name
            file_hash = compute_file_hash(path)
        else:
            fname = filename or getattr(source, "name", "uploaded_document")
            # For stream, read bytes for hash then seek back
            content = source.read() if hasattr(source, "read") else b""
            file_hash = compute_file_hash(content)
            if hasattr(source, "seek"):
                source.seek(0)

        # Check duplicate
        if self.vector_store.is_document_already_indexed(fname, file_hash):
            logger.info(f"Document '{fname}' with identical hash already indexed. Skipping.")
            return {
                "document_name": fname,
                "status": "skipped",
                "message": "Document already indexed with identical content.",
                "chunks_added": 0
            }

        # Step 1: Load and parse document
        pages = DocumentLoader.load_document(source, filename=fname)
        if not pages:
            return {
                "document_name": fname,
                "status": "empty",
                "message": "No text content found in document.",
                "chunks_added": 0
            }

        # Step 2: Chunk document
        chunks = self.chunker.chunk_documents(pages)
        if not chunks:
            return {
                "document_name": fname,
                "status": "empty_chunks",
                "message": "Chunking produced no content.",
                "chunks_added": 0
            }

        # Step 3: Embed chunks
        chunk_texts = [c["text"] for c in chunks]
        embeddings = self.embedding_service.embed_texts(chunk_texts)

        # Step 4: Add to vector store and metadata
        added_count = self.vector_store.add_chunks(chunks, embeddings, doc_hash=file_hash)

        return {
            "document_name": fname,
            "status": "success",
            "message": f"Successfully indexed {added_count} chunks.",
            "chunks_added": added_count
        }

    def ingest_multiple_documents(
        self,
        sources: List[Union[str, Path, BinaryIO]],
        filenames: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Ingests a batch of documents."""
        results = []
        for idx, src in enumerate(sources):
            name = filenames[idx] if filenames and idx < len(filenames) else ""
            res = self.ingest_document(src, filename=name)
            results.append(res)
        return results

    def ingest_sample_documents(self) -> List[Dict[str, Any]]:
        """Ingests all files from data/sample_documents/."""
        if not SAMPLE_DOCS_DIR.exists():
            return []

        doc_files = [
            f for f in SAMPLE_DOCS_DIR.iterdir()
            if f.is_file() and f.suffix.lower() in DocumentLoader.SUPPORTED_EXTENSIONS
        ]
        doc_files.sort(key=lambda x: x.name)
        return self.ingest_multiple_documents(doc_files)

    def rebuild_knowledge_base(self, include_samples: bool = True) -> Dict[str, Any]:
        """Clears the FAISS index and rebuilds from scratch."""
        self.vector_store.clear()
        results = []
        if include_samples:
            results = self.ingest_sample_documents()
        return {
            "status": "rebuilt",
            "total_chunks": self.vector_store.total_chunks,
            "total_documents": self.vector_store.total_documents,
            "ingestion_results": results
        }

    def answer_question(
        self,
        question: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Full RAG pipeline for answering an employee question:
        question -> retrieval -> relevance evaluation -> generation -> source attribution
        """
        # Step 1: Retrieve context
        retrieval_result = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            threshold=threshold
        )

        # Step 2: Generate grounded answer
        gen_result = self.generator.generate_response(retrieval_result)

        # Return consolidated result
        return {
            "question": question,
            "answer": gen_result["answer"],
            "sources": gen_result["sources"],
            "sources_formatted": gen_result["sources_formatted"],
            "is_fallback": gen_result["is_fallback"],
            "chunks_used": gen_result["chunks_used"],
            "raw_candidates": retrieval_result["raw_candidates"],
            "top_score": retrieval_result["top_score"],
            "threshold": retrieval_result["threshold"],
            "is_relevant": retrieval_result["is_relevant"],
            "latency_ms": retrieval_result["latency_ms"],
            "debug_summary": retrieval_result["debug_summary"]
        }
