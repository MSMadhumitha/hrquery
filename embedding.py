"""
Embedding Module for HRQuery RAG.
Generates dense semantic vector embeddings using the official Google GenAI SDK
(model: text-embedding-004, 768 dimensions).
Includes L2-normalization so FAISS Inner Product equals Cosine Similarity.
Provides a deterministic offline fallback vectorizer for headless automated testing.
"""

import logging
import os
import numpy as np
from typing import List, Optional
from google import genai
from config import EMBEDDING_MODEL, EMBEDDING_DIMENSION, GEMINI_API_KEY

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating semantic vector embeddings for text chunks and queries.
    Uses Google GenAI text-embedding-004 by default, with deterministic fallback for tests.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = EMBEDDING_MODEL):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "") or GEMINI_API_KEY
        self.model = model
        self.dimension = EMBEDDING_DIMENSION
        self._client = None

        if self.api_key and self.api_key != "your_gemini_api_key_here":
            try:
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Google GenAI Client: {e}. Falling back to local vectorizer.")

    @property
    def is_live(self) -> bool:
        """True if authenticated with Google GenAI API."""
        return self._client is not None

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embeds a list of text strings into an (N, 768) float32 numpy array.
        All output vectors are L2-normalized so dot products equal cosine similarities.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        if self._client:
            try:
                return self._embed_with_gemini(texts)
            except Exception as e:
                logger.error(f"Gemini embedding API call failed: {e}. Using fallback vectorizer.")

        return self._embed_offline_fallback(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embeds a single search query into a (1, 768) float32 numpy array.
        Uses identical embedding method as chunk documents.
        """
        vectors = self.embed_texts([query])
        return vectors

    def _embed_with_gemini(self, texts: List[str]) -> np.ndarray:
        """Calls official Google GenAI embed_content API in batches."""
        # Batch size for text-embedding-004 (Google API supports up to 100 texts per batch)
        batch_size = 50
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._client.models.embed_content(
                model=self.model,
                contents=batch
            )
            for item in response.embeddings:
                all_embeddings.append(item.values)

        matrix = np.array(all_embeddings, dtype=np.float32)
        # Normalize each vector to unit length (L2 norm)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_matrix = matrix / norms
        return normalized_matrix

    def _embed_offline_fallback(self, texts: List[str]) -> np.ndarray:
        """
        Deterministic, offline semantic-style hash vectorizer for testing without API keys.
        Creates continuous 768-dimensional representations based on n-grams and tokens,
        ensuring unit-normalized vectors where semantically overlapping texts produce high cosine similarity.
        """
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        
        for row_idx, text in enumerate(texts):
            clean = text.lower().strip()
            words = clean.split()
            if not words:
                matrix[row_idx, 0] = 1.0
                continue
                
            vec = np.zeros(self.dimension, dtype=np.float32)
            
            # Encode words and character 3-grams
            tokens = words + [clean[i : i + 3] for i in range(len(clean) - 2)]
            for token in tokens:
                # Deterministic polynomial hash
                h = 0
                for char in token:
                    h = (h * 31 + ord(char)) & 0xFFFFFFFF
                
                dim_idx = h % self.dimension
                sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
                weight = 1.0 if len(token) > 2 else 0.5
                vec[dim_idx] += sign * weight

            # Unit normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            else:
                vec[0] = 1.0
            matrix[row_idx] = vec

        return matrix
