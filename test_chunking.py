"""
Unit tests for document chunking module (ingestion/chunker.py).
Validates size limits, overlap, paragraph boundary preservation, and metadata.
"""

import pytest
from ingestion.chunker import TextChunker


def test_chunker_initialization():
    chunker = TextChunker(chunk_size=500, chunk_overlap=100)
    assert chunker.chunk_size == 500
    assert chunker.chunk_overlap == 100

    # Overlap must be strictly less than chunk_size
    with pytest.raises(ValueError):
        TextChunker(chunk_size=200, chunk_overlap=250)


def test_chunker_preserves_metadata():
    chunker = TextChunker(chunk_size=400, chunk_overlap=50)
    pages = [
        {
            "document_name": "Leave_Policy.pdf",
            "page_number": 4,
            "source_type": "pdf",
            "text": "Employees are entitled to 12 days of Casual Leave annually. Requests must be submitted 48 hours in advance."
        }
    ]

    chunks = chunker.chunk_documents(pages)
    assert len(chunks) >= 1
    c = chunks[0]
    assert c["document_name"] == "Leave_Policy.pdf"
    assert c["page_number"] == 4
    assert c["source_type"] == "pdf"
    assert "leave_policy_pdf" in c["chunk_id"]
    assert "Casual Leave" in c["text"]


def test_chunker_paragraph_boundary_awareness():
    chunker = TextChunker(chunk_size=300, chunk_overlap=50)
    p1 = "Paragraph 1: Welcome to NovaTech Solutions Inc. We build world-class AI systems for enterprise customers."
    p2 = "Paragraph 2: All full-time employees are provided health coverage and wellness stipends."
    p3 = "Paragraph 3: Attendance is strictly monitored through our digital badge system."
    combined = f"{p1}\n\n{p2}\n\n{p3}"

    pages = [{"document_name": "Test_Doc.txt", "page_number": 1, "source_type": "txt", "text": combined}]
    chunks = chunker.chunk_documents(pages)

    # Chunks should not break paragraphs mid-word or arbitrarily when they fit within chunk_size
    assert len(chunks) >= 1
    # Check that paragraph structure is maintained
    found_p1 = any("Paragraph 1" in c["text"] for c in chunks)
    found_p2 = any("Paragraph 2" in c["text"] for c in chunks)
    assert found_p1 and found_p2


def test_chunker_empty_input():
    chunker = TextChunker(chunk_size=500, chunk_overlap=100)
    chunks = chunker.chunk_documents([])
    assert chunks == []

    chunks_empty_text = chunker.chunk_documents([
        {"document_name": "Empty.txt", "page_number": 1, "source_type": "txt", "text": "   "}
    ])
    assert chunks_empty_text == []
