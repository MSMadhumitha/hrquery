"""
Retriever Module for HRQuery RAG.
Coordinates query embedding, FAISS similarity search, relevance thresholding,
and debug diagnostics.
"""

import time
import logging
from typing import List, Dict, Any, Tuple
from config import DEFAULT_TOP_K, DEFAULT_RELEVANCE_THRESHOLD
from ingestion.embedding import EmbeddingService
from retrieval.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class HRQueryRetriever:
    """
    Executes semantic retrieval across indexed policy chunks.
    Enforces relevance thresholding and formats debug inspection payloads.
    """

    def __init__(
        self,
        vector_store: VectorStoreManager,
        embedding_service: EmbeddingService,
        top_k: int = DEFAULT_TOP_K,
        relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        threshold: float = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end retrieval for a query string:
        Query -> Query Embedding -> FAISS Top-K -> Relevance Check -> Structured Result

        Returns a dictionary:
        {
            "query": "How many casual leaves...",
            "chunks": [ {...chunk_meta, "similarity_score": 0.82}, ... ],
            "is_relevant": True/False,
            "top_score": 0.82,
            "threshold": 0.38,
            "latency_ms": 14.2,
            "debug_summary": "Retrieved 5 chunks. Top score: 0.82 (>= 0.38). Document: Leave_Policy.pdf"
        }
        """
        start_time = time.perf_counter()
        k = top_k or self.top_k
        min_threshold = threshold if threshold is not None else self.relevance_threshold

        if not query or not query.strip():
            return {
                "query": query,
                "chunks": [],
                "is_relevant": False,
                "top_score": 0.0,
                "threshold": min_threshold,
                "latency_ms": 0.0,
                "debug_summary": "Empty query provided."
            }

        if self.vector_store.total_chunks == 0:
            return {
                "query": query,
                "chunks": [],
                "is_relevant": False,
                "top_score": 0.0,
                "threshold": min_threshold,
                "latency_ms": 0.0,
                "debug_summary": "Vector store is empty. No documents indexed."
            }

        # Step 1: Embed query using identical model and normalization
        query_vector = self.embedding_service.embed_query(query)

        # Step 2: Query FAISS index for top-k nearest neighbors
        retrieved_chunks = self.vector_store.search(query_vector, top_k=k)

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Step 3: Evaluate relevance against threshold
        top_score = retrieved_chunks[0]["similarity_score"] if retrieved_chunks else 0.0
        is_relevant = bool(retrieved_chunks and (top_score >= min_threshold))

        # Filter chunks that are sufficiently close to be useful as LLM context
        # Even if top_score >= threshold, we discard chunks with negative or very low similarity
        filtered_chunks = [
            c for c in retrieved_chunks if c["similarity_score"] >= max(0.20, min_threshold - 0.15)
        ] if is_relevant else []

        debug_summary = (
            f"Retrieved {len(retrieved_chunks)} candidate chunks in {latency_ms}ms. "
            f"Top similarity: {top_score:.4f} "
            f"(Threshold: {min_threshold:.2f} -> {'PASSED' if is_relevant else 'FALLBACK'})."
        )

        return {
            "query": query,
            "chunks": filtered_chunks if is_relevant else retrieved_chunks,
            "raw_candidates": retrieved_chunks,
            "is_relevant": is_relevant,
            "top_score": top_score,
            "threshold": min_threshold,
            "latency_ms": latency_ms,
            "debug_summary": debug_summary
        }
