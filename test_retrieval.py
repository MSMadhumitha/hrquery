"""
Integration and unit tests for document loading, vector storage, and retrieval accuracy (retrieval/retriever.py).
"""

import pytest
from pathlib import Path
from config import SAMPLE_DOCS_DIR
from ingestion.document_loader import DocumentLoader
from ingestion.chunker import TextChunker
from ingestion.embedding import EmbeddingService
from retrieval.vector_store import VectorStoreManager
from retrieval.retriever import HRQueryRetriever


def test_document_loader_formats():
    """Verify PDF, DOCX, and TXT loaders extract text and metadata."""
    # 1. TXT loader
    txt_path = SAMPLE_DOCS_DIR / "Resignation_Policy.txt"
    assert txt_path.exists(), "Sample Resignation_Policy.txt must exist"
    txt_pages = DocumentLoader.load_document(txt_path)
    assert len(txt_pages) == 1
    assert "RESIGNATION" in txt_pages[0]["text"].upper()
    assert txt_pages[0]["source_type"] == "txt"

    # 2. PDF loader
    pdf_path = SAMPLE_DOCS_DIR / "Leave_Policy.pdf"
    assert pdf_path.exists(), "Sample Leave_Policy.pdf must exist"
    pdf_pages = DocumentLoader.load_document(pdf_path)
    assert len(pdf_pages) >= 2, "Leave Policy should have multiple pages"
    assert pdf_pages[0]["source_type"] == "pdf"
    assert pdf_pages[0]["page_number"] == 1
    assert pdf_pages[1]["page_number"] == 2

    # 3. DOCX loader
    docx_path = SAMPLE_DOCS_DIR / "Attendance_Policy.docx"
    assert docx_path.exists(), "Sample Attendance_Policy.docx must exist"
    docx_pages = DocumentLoader.load_document(docx_path)
    assert len(docx_pages) >= 1
    assert "ATTENDANCE" in docx_pages[0]["text"].upper()
    assert docx_pages[0]["source_type"] == "docx"


def test_vector_store_persistence(tmp_path):
    """Test FAISS Index and metadata persistence and reloading."""
    index_file = tmp_path / "test.faiss"
    meta_file = tmp_path / "test_meta.json"

    vs1 = VectorStoreManager(index_path=index_file, metadata_path=meta_file, dimension=768)
    embedder = EmbeddingService()

    chunks = [
        {
            "chunk_id": "c1",
            "document_name": "Test_Doc.pdf",
            "page_number": 1,
            "source_type": "pdf",
            "text": "NovaTech provides comprehensive healthcare coverage."
        }
    ]
    vectors = embedder.embed_texts([chunks[0]["text"]])
    vs1.add_chunks(chunks, vectors, doc_hash="fakehash123")

    assert vs1.total_chunks == 1
    assert vs1.is_document_already_indexed("Test_Doc.pdf", "fakehash123") is True

    # Reload into a fresh manager instance from disk
    vs2 = VectorStoreManager(index_path=index_file, metadata_path=meta_file, dimension=768)
    assert vs2.total_chunks == 1
    assert vs2.metadata["chunks"][0]["document_name"] == "Test_Doc.pdf"


def test_retrieval_accuracy_on_hr_queries():
    """
    Retrieval quality is 20% of the hackathon score.
    Verify that domain questions accurately retrieve the expected policy documents.
    """
    vector_store = VectorStoreManager()
    embedder = EmbeddingService()
    retriever = HRQueryRetriever(vector_store=vector_store, embedding_service=embedder, top_k=3)

    # 1. Casual Leaves -> Leave_Policy.pdf
    res1 = retriever.retrieve("How many casual leaves are employees entitled to?")
    assert res1["is_relevant"] is True
    assert len(res1["chunks"]) > 0
    top_doc1 = res1["chunks"][0]["document_name"]
    assert "Leave_Policy" in top_doc1

    # 2. Resignation notice -> Resignation_Policy.txt
    res2 = retriever.retrieve("What is the resignation notice period?")
    assert res2["is_relevant"] is True
    top_doc2 = res2["chunks"][0]["document_name"]
    assert "Resignation_Policy" in top_doc2

    # 3. Work from home health issue -> Work_From_Home_Policy.pdf
    res3 = retriever.retrieve("Can an employee work from home during a temporary health issue?")
    assert res3["is_relevant"] is True
    top_doc3 = res3["chunks"][0]["document_name"]
    assert "Work_From_Home" in top_doc3

    # 4. Travel expenses -> Travel_and_Expense_Policy.pdf
    res4 = retriever.retrieve("How are travel expenses reimbursed?")
    assert res4["is_relevant"] is True
    top_doc4 = res4["chunks"][0]["document_name"]
    assert "Travel_and_Expense" in top_doc4
