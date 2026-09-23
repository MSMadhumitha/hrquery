"""
FAISS Vector Store Manager for HRQuery RAG.
Stores and searches dense semantic vectors using FAISS IndexFlatIP (Cosine Similarity).
Handles persistence, metadata synchronization, and duplicate document guards.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import faiss
from config import (
    FAISS_INDEX_PATH,
    METADATA_STORE_PATH,
    EMBEDDING_DIMENSION,
    VECTOR_STORE_DIR
)

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages persistent FAISS Index and synchronized chunk metadata.
    """

    def __init__(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        metadata_path: Path = METADATA_STORE_PATH,
        dimension: int = EMBEDDING_DIMENSION
    ):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.dimension = dimension
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: Dict[str, Any] = {
            "chunks": [],
            "document_hashes": {},
            "dimension": self.dimension,
            "last_updated": None
        }
        self.load_or_initialize()

    def load_or_initialize(self) -> None:
        """Loads index and metadata from disk if available, otherwise initializes empty."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        if self.index_path.exists() and self.metadata_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                logger.info(
                    f"Loaded FAISS index with {self.index.ntotal} vectors "
                    f"and {len(self.metadata['chunks'])} chunk metadata records."
                )
                return
            except Exception as e:
                logger.warning(f"Error loading existing vector store: {e}. Reinitializing new index.")

        self._init_empty()

    def _init_empty(self) -> None:
        """Initializes a new empty FAISS IndexFlatIP."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = {
            "chunks": [],
            "document_hashes": {},
            "dimension": self.dimension,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }

    def save(self) -> None:
        """Persists the FAISS index and metadata to disk."""
        if self.index is None:
            return
        
        self.metadata["last_updated"] = datetime.now(timezone.utc).isoformat()
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved FAISS index ({self.index.ntotal} vectors) to {self.index_path}")

    def clear(self) -> None:
        """Resets the vector store and removes disk artifacts."""
        self._init_empty()
        if self.index_path.exists():
            self.index_path.unlink()
        if self.metadata_path.exists():
            self.metadata_path.unlink()
        logger.info("Cleared vector store index and metadata.")

    def is_document_already_indexed(self, doc_name: str, file_hash: str) -> bool:
        """Checks if a document with the same name and content hash has already been indexed."""
        stored_hash = self.metadata.get("document_hashes", {}).get(doc_name)
        return stored_hash == file_hash

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray,
        doc_hash: Optional[str] = None
    ) -> int:
        """
        Adds new chunk vectors and metadata to the index.
        embeddings must be of shape (len(chunks), dimension) and float32.
        """
        if len(chunks) == 0:
            return 0

        if embeddings.shape[0] != len(chunks):
            raise ValueError(f"Mismatch between chunks count ({len(chunks)}) and embeddings count ({embeddings.shape[0]}).")

        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Expected embedding dimension {self.dimension}, got {embeddings.shape[1]}.")

        # Ensure float32 format for FAISS
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)

        # Add vectors to FAISS index
        self.index.add(vectors)

        # Append chunk metadata
        for chunk in chunks:
            self.metadata["chunks"].append({
                "chunk_id": chunk.get("chunk_id", ""),
                "document_name": chunk.get("document_name", ""),
                "page_number": chunk.get("page_number", 1),
                "source_type": chunk.get("source_type", ""),
                "text": chunk.get("text", "")
            })

        # Track document hash if provided
        if doc_hash and chunks:
            doc_name = chunks[0].get("document_name", "")
            if doc_name:
                self.metadata["document_hashes"][doc_name] = doc_hash

        self.save()
        return len(chunks)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Searches the FAISS index for the top_k most similar chunks.
        query_vector must be of shape (1, dimension) and float32 (L2-normalized).
        
        Returns a list of dicts with chunk metadata + similarity_score.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        top_k = min(top_k, self.index.ntotal)
        query = np.ascontiguousarray(query_vector, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # FAISS search returns similarity scores (Inner Product) and indices
        scores, indices = self.index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata["chunks"]):
                continue

            chunk_meta = dict(self.metadata["chunks"][idx])
            # For normalized vectors with IndexFlatIP, score is Cosine Similarity in [-1, 1]
            chunk_meta["similarity_score"] = float(score)
            results.append(chunk_meta)

        return results

    @property
    def total_chunks(self) -> int:
        return self.index.ntotal if self.index else 0

    @property
    def total_documents(self) -> int:
        unique_docs = {c.get("document_name") for c in self.metadata.get("chunks", [])}
        return len(unique_docs)
