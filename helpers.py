import hashlib
import re
from typing import List, Dict, Any, Tuple
from pathlib import Path


def compute_file_hash(data: bytes | str | Path) -> str:
    """
    Computes an MD5 hex digest for binary content or a file path.
    Used for duplicate document detection in vector storage.
    """
    hasher = hashlib.md5()
    if isinstance(data, (str, Path)):
        path = Path(data)
        if path.is_file():
            with open(path, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            return hasher.hexdigest()
        else:
            hasher.update(str(data).encode("utf-8"))
    elif isinstance(data, bytes):
        hasher.update(data)
    else:
        hasher.update(str(data).encode("utf-8"))
    return hasher.hexdigest()


def clean_text(text: str) -> str:
    """
    Cleans raw document text:
    - Normalizes carriage returns and multiple consecutive spaces.
    - Preserves paragraph separations (double newlines).
    - Removes non-printable characters.
    """
    if not text:
        return ""
    
    # Replace carriage returns with standard newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Remove control characters except standard whitespace
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    
    # Collapse 3 or more newlines into 2 (paragraph boundary)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # Normalize excessive horizontal whitespace within lines
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    
    return "\n".join(lines).strip()


def deduplicate_sources(chunks: List[Dict[str, Any]]) -> List[Tuple[str, Any]]:
    """
    Extracts unique (document_name, page_number) pairs from retrieved chunks
    preserving the original retrieval ranking order.
    """
    seen = set()
    unique_sources = []
    
    for chunk in chunks:
        doc_name = chunk.get("document_name", "Unknown Document")
        page_num = chunk.get("page_number")
        key = (doc_name, page_num)
        if key not in seen:
            seen.add(key)
            unique_sources.append(key)
            
    return unique_sources


def format_sources(chunks: List[Dict[str, Any]]) -> str:
    """
    Formats source attributions strictly according to the hackathon specification:
    Sources:
    1. Leave Policy.pdf — Page 4
    2. Employee Handbook.pdf — Page 18
    
    If page number is unknown or not applicable, renders 'Page unavailable'.
    Never fabricates citations.
    """
    sources = deduplicate_sources(chunks)
    if not sources:
        return ""
    
    lines = ["**Sources:**"]
    for idx, (doc_name, page_num) in enumerate(sources, start=1):
        if page_num is not None and str(page_num).strip() and str(page_num) != "0":
            page_str = f"Page {page_num}"
        else:
            page_str = "Page unavailable"
        lines.append(f"{idx}. {doc_name} — {page_str}")
        
    return "\n".join(lines)
