"""
Tests for Grounding Guardrails, Relevance Thresholding, and Source Citations (generation/llm.py).
"""

import pytest
from config import GROUNDING_FALLBACK_MESSAGE
from generation.llm import GeminiRAGGenerator
from utils.helpers import format_sources, deduplicate_sources
from pipeline import HRQueryPipeline


def test_grounding_fallback_exact_string():
    """Verify exact fallback string on out-of-scope and weak queries."""
    generator = GeminiRAGGenerator()

    # Case 1: is_relevant is False
    irrelevant_result = {
        "query": "What is the price of Bitcoin in 2026?",
        "chunks": [],
        "is_relevant": False,
        "top_score": 0.12,
        "threshold": 0.38,
        "debug_summary": "Low score."
    }
    resp = generator.generate_response(irrelevant_result)
    assert resp["is_fallback"] is True
    assert resp["answer"] == GROUNDING_FALLBACK_MESSAGE
    assert resp["sources"] == []
    assert resp["sources_formatted"] == ""


def test_out_of_scope_questions_trigger_fallback():
    """
    Hackathon requirement:
    6. 'What is the salary of a software engineer with three years of experience?' -> must trigger fallback
    7. 'What is the cafeteria menu?' -> must trigger fallback
    """
    pipeline = HRQueryPipeline()

    # Question 6: Salary inquiry (not in policy documents)
    q6 = "What is the salary of a software engineer with three years of experience?"
    res6 = pipeline.answer_question(q6)
    assert res6["is_fallback"] is True
    assert res6["answer"] == GROUNDING_FALLBACK_MESSAGE
    assert res6["sources"] == []

    # Question 7: Cafeteria menu
    q7 = "What is the cafeteria menu?"
    res7 = pipeline.answer_question(q7)
    assert res7["is_fallback"] is True
    assert res7["answer"] == GROUNDING_FALLBACK_MESSAGE


def test_source_attribution_formatting():
    """Verify source formatting matches required format without fabricated sources."""
    chunks = [
        {"document_name": "Leave_Policy.pdf", "page_number": 4},
        {"document_name": "Leave_Policy.pdf", "page_number": 4},  # Duplicate page
        {"document_name": "Employee_Handbook.pdf", "page_number": 18},
        {"document_name": "Resignation_Policy.txt", "page_number": 0},  # Unknown page
    ]

    unique_sources = deduplicate_sources(chunks)
    assert len(unique_sources) == 3
    assert ("Leave_Policy.pdf", 4) in unique_sources
    assert ("Employee_Handbook.pdf", 18) in unique_sources

    formatted = format_sources(chunks)
    assert "**Sources:**" in formatted
    assert "1. Leave_Policy.pdf — Page 4" in formatted
    assert "2. Employee_Handbook.pdf — Page 18" in formatted
    assert "3. Resignation_Policy.txt — Page unavailable" in formatted
