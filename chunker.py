"""
Chunking Module for HRQuery RAG.

=============================================================================
WHY CHUNKING MATTERS FOR RETRIEVAL QUALITY (AND WHY PARAGRAPH-AWARE MATTERS):
=============================================================================
1. Semantic Granularity & Context Dilution:
   Embedding models encode an entire text block into a single dense vector.
   If a chunk is too large (e.g., an entire policy document), the vector represents
   an average of dozens of distinct topics. A specific query like "casual leave notice period"
   gets drowned out in the vector representation (the "needle-in-a-haystack" dilution).
   Conversely, if chunks are too small (e.g., single sentences), the chunk lacks the
   necessary context, exceptions, and conditions (e.g., "Requires 48 hours notice" without knowing it applies to Casual Leave).

2. Paragraph & Section Boundary Preservation:
   Human authors structure documents into paragraphs to group related thoughts.
   Arbitrarily chopping text at a fixed character count often splits sentences or
   detaches a policy rule from its eligibility condition. Preserving paragraph and
   sentence boundaries maintains logical coherence, producing significantly cleaner
   and more accurate semantic vector representations.

3. Sliding Window Overlap:
   Overlap ensures boundary protection. If a critical concept or rule spans the boundary
   between two adjacent chunks, overlap guarantees that at least one chunk captures
   the complete context, preventing false-negative retrieval failures.
"""

import re
from typing import List, Dict, Any
from config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


class TextChunker:
    """
    Paragraph-aware text chunker with configurable size and overlap.
    Preserves document metadata and assigns unique chunk IDs.
    """

    def __init__(
        self, 
        chunk_size: int = DEFAULT_CHUNK_SIZE, 
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(
        self, 
        pages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Splits a list of document pages/sections into structured chunks.
        
        Input page format:
        {
            "document_name": "Leave_Policy.pdf",
            "page_number": 1,
            "source_type": "pdf",
            "text": "..."
        }
        
        Output chunk format:
        {
            "chunk_id": "leave_policy_pdf_p01_c001",
            "document_name": "Leave_Policy.pdf",
            "page_number": 1,
            "source_type": "pdf",
            "text": "Employees are entitled to...",
            "token_estimate": 142
        }
        """
        all_chunks = []
        chunk_counter = 1

        for page in pages:
            raw_text = page.get("text", "").strip()
            if not raw_text:
                continue

            doc_name = page.get("document_name", "unknown")
            page_num = page.get("page_number", 1)
            source_type = page.get("source_type", "txt")
            
            # Create a sanitized base for chunk IDs
            clean_name = re.sub(r"[^a-zA-Z0-9]+", "_", doc_name.lower()).strip("_")
            
            # Generate chunks for this specific page/section
            page_text_chunks = self._split_text(raw_text)
            
            for p_chunk in page_text_chunks:
                chunk_id = f"{clean_name}_p{page_num:02d}_c{chunk_counter:03d}"
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "document_name": doc_name,
                    "page_number": page_num,
                    "source_type": source_type,
                    "text": p_chunk,
                    "token_estimate": max(1, len(p_chunk) // 4)
                })
                chunk_counter += 1

        return all_chunks

    def _split_text(self, text: str) -> List[str]:
        """
        Splits text into chunks respecting paragraph, line, and sentence boundaries.
        """
        if len(text) <= self.chunk_size:
            return [text]

        # Break text down into structural units: paragraphs first
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        
        # If no double-newlines, break by single newlines
        if len(paragraphs) <= 1:
            paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

        # Further subdivide any paragraph that exceeds chunk_size into sentences
        units = []
        for p in paragraphs:
            if len(p) <= self.chunk_size:
                units.append(p)
            else:
                sentences = self._split_sentences(p)
                for s in sentences:
                    if len(s) <= self.chunk_size:
                        units.append(s)
                    else:
                        # Hard split only as a last resort if a single sentence exceeds chunk_size
                        units.extend(self._hard_split(s, self.chunk_size))

        # Recombine units using a sliding window with overlap
        chunks = []
        current_chunk = []
        current_length = 0

        for unit in units:
            unit_len = len(unit)
            # Check if adding this unit exceeds target chunk_size
            if current_chunk and (current_length + unit_len + 2 > self.chunk_size):
                chunk_str = "\n\n".join(current_chunk).strip()
                if chunk_str:
                    chunks.append(chunk_str)

                # Implement overlap: retain the tail of current_chunk up to chunk_overlap characters
                overlap_units = []
                overlap_len = 0
                for prev_unit in reversed(current_chunk):
                    if overlap_len + len(prev_unit) <= self.chunk_overlap:
                        overlap_units.insert(0, prev_unit)
                        overlap_len += len(prev_unit)
                    else:
                        break
                
                current_chunk = overlap_units
                current_length = sum(len(u) for u in current_chunk) + (len(current_chunk) * 2 if current_chunk else 0)

            current_chunk.append(unit)
            current_length += unit_len + 2

        if current_chunk:
            final_str = "\n\n".join(current_chunk).strip()
            if final_str and (not chunks or final_str != chunks[-1]):
                chunks.append(final_str)

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Splits a block of text into sentences using regex boundary detection."""
        sentence_end = re.compile(r'(?<=[.!?])\s+')
        sentences = sentence_end.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _hard_split(self, text: str, size: int) -> List[str]:
        """Splits long strings without natural boundaries at word spaces."""
        words = text.split(" ")
        chunks = []
        cur = []
        cur_len = 0
        for w in words:
            if cur_len + len(w) + 1 > size and cur:
                chunks.append(" ".join(cur))
                cur = [w]
                cur_len = len(w)
            else:
                cur.append(w)
                cur_len += len(w) + 1
        if cur:
            chunks.append(" ".join(cur))
        return chunks
