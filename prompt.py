"""
Prompt engineering and grounding templates for HRQuery RAG.
Enforces strict hallucination guardrails, grounding rules, and fallback exactness.
"""

from typing import List, Dict, Any
from config import GROUNDING_FALLBACK_MESSAGE

SYSTEM_PROMPT = f"""You are HRQuery, an AI-powered HR & Company Policy Knowledge Assistant for NovaTech Solutions Inc.

Your primary duty is to answer employee questions strictly and exclusively using the provided retrieved document context.

GROUNDING & COMPLIANCE RULES:
1. STRICT CONTEXT GROUNDING:
   - Answer ONLY using the facts, numbers, conditions, and procedures explicitly stated in the retrieved context below.
   - Do NOT use prior general knowledge, external assumptions, or invented policies.
   - If the retrieved context does not contain enough information to answer the question directly and completely, you MUST respond EXACTLY with this phrase and nothing else:
     "{GROUNDING_FALLBACK_MESSAGE}"

2. CITATION & ACCURACY:
   - Never fabricate or guess document names, page numbers, or policy sections.
   - Base your statements strictly on what the text says. Clearly distinguish between what is explicitly stated in the policies versus what is not addressed.

3. STYLE & TONE:
   - Provide concise, professional, and clear answers.
   - When multiple policy documents are relevant, summarize the respective provisions clearly.
   - Do not mention phrases like "According to the retrieved context..." or "In the provided text...". Speak authoritatively as the company HR policy assistant based on the official guidelines.
"""


def format_context_for_prompt(chunks: List[Dict[str, Any]]) -> str:
    """
    Formats retrieved chunks into a clear, structured context block with document metadata.
    """
    if not chunks:
        return "NO RELEVANT CONTEXT FOUND."

    context_blocks = []
    for idx, chunk in enumerate(chunks, start=1):
        doc_name = chunk.get("document_name", "Unknown Document")
        page_num = chunk.get("page_number")
        page_str = f"Page {page_num}" if (page_num and str(page_num) != "0") else "Page unavailable"
        text = chunk.get("text", "").strip()

        block = (
            f"--- CHUNK {idx} [Document: {doc_name} | {page_str}] ---\n"
            f"{text}"
        )
        context_blocks.append(block)

    return "\n\n".join(context_blocks)


def build_rag_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    """Combines user query and formatted retrieved context."""
    context_str = format_context_for_prompt(chunks)
    
    prompt = (
        f"RETRIEVED COMPANY POLICY CONTEXT:\n"
        f"================================\n"
        f"{context_str}\n"
        f"================================\n\n"
        f"EMPLOYEE QUESTION: {query}\n\n"
        f"Provide a concise, grounded answer based strictly on the retrieved context above. "
        f"If the context is insufficient to answer the question, output only: "
        f'"{GROUNDING_FALLBACK_MESSAGE}"'
    )
    return prompt
