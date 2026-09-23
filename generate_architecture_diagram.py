"""
Generates high-resolution architecture diagram PNG for HRQuery RAG application.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUTPUT_PATH = Path(__file__).resolve().parent / "architecture.png"


def create_diagram():
    width, height = 1200, 750
    img = Image.new("RGB", (width, height), color="#F8FAFC")
    draw = ImageDraw.Draw(img)

    # Load default fonts
    try:
        font_title = ImageFont.truetype("arialbd.ttf", 26)
        font_section = ImageFont.truetype("arialbd.ttf", 18)
        font_box_title = ImageFont.truetype("arialbd.ttf", 15)
        font_text = ImageFont.truetype("arial.ttf", 12)
        font_tag = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font_title = ImageFont.load_default()
        font_section = font_title
        font_box_title = font_title
        font_text = font_title
        font_tag = font_title

    # Header Bar
    draw.rectangle([(0, 0), (width, 80)], fill="#1E3A8A")
    draw.text((40, 16), "HRQuery — Enterprise RAG Architecture Pipeline", fill="#FFFFFF", font=font_title)
    draw.text((40, 50), "Document Ingestion, FAISS Similarity Search, Strict Grounding & Gemini Generation", fill="#BFDBFE", font=font_tag)

    def draw_card(x, y, w, h, title, subtitle, details, bg_color="#FFFFFF", border_color="#CBD5E1", header_color="#1E293B"):
        # Shadow
        draw.rounded_rectangle([(x + 2, y + 2), (x + w + 2, y + h + 2)], radius=8, fill="#E2E8F0")
        # Body
        draw.rounded_rectangle([(x, y), (x + w, y + h)], radius=8, fill=bg_color, outline=border_color, width=2)
        # Title
        draw.text((x + 16, y + 14), title, fill=header_color, font=font_box_title)
        draw.text((x + 16, y + 36), subtitle, fill="#64748B", font=font_tag)
        # Divider
        draw.line([(x + 16, y + 54), (x + w - 16, y + 54)], fill="#E2E8F0", width=1)
        # Details
        cy = y + 64
        for d in details:
            draw.text((x + 16, cy), f"• {d}", fill="#334155", font=font_text)
            cy += 18

    # SECTION 1: INGESTION PIPELINE
    draw.text((40, 105), "1. INGESTION PIPELINE (Offline / Real-Time Upload)", fill="#0F172A", font=font_section)

    draw_card(40, 140, 250, 170, "Document Ingestion", "PDF, DOCX, TXT", [
        "pypdf: page-aware parser",
        "python-docx: paragraphs & tables",
        "UTF-8 text decoder",
        "MD5 file hash deduplication"
    ], bg_color="#EFF6FF", border_color="#93C5FD", header_color="#1D4ED8")

    draw_card(330, 140, 250, 170, "Paragraph Chunker", "Semantic Windows", [
        "1000 char target size",
        "150 char sliding overlap",
        "Paragraph (\\n\\n) boundary aware",
        "Preserves doc name & page num"
    ], bg_color="#F0FDF4", border_color="#86EFAC", header_color="#15803D")

    draw_card(620, 140, 250, 170, "Embedding Engine", "Google GenAI API", [
        "Model: text-embedding-004",
        "768-dimensional dense vectors",
        "L2-normalization: ||v|| = 1.0",
        "Deterministic test fallback"
    ], bg_color="#FAF5FF", border_color="#D8B4FE", header_color="#7E22CE")

    draw_card(910, 140, 250, 170, "Vector Storage", "FAISS + Metadata", [
        "FAISS IndexFlatIP (Cosine)",
        "Persistent index.faiss on disk",
        "Persistent metadata.json store",
        "Rebuilding & instant reload"
    ], bg_color="#FEF3C7", border_color="#FCD34D", header_color="#B45309")

    # Arrows for Ingestion
    def draw_arrow(x1, y1, x2, y2, color="#64748B"):
        draw.line([(x1, y1), (x2, y2)], fill=color, width=3)
        draw.polygon([(x2, y2), (x2 - 8, y2 - 5), (x2 - 8, y2 + 5)], fill=color)

    draw_arrow(292, 225, 328, 225)
    draw_arrow(582, 225, 618, 225)
    draw_arrow(872, 225, 908, 225)

    # SECTION 2: QUERY & GENERATION PIPELINE
    draw.text((40, 345), "2. QUERY & GENERATION PIPELINE (Runtime)", fill="#0F172A", font=font_section)

    draw_card(40, 380, 250, 170, "Employee Query", "User Interaction", [
        "Streamlit interactive input",
        "Sample policy test buttons",
        "Identical text-embedding-004",
        "Normalized query vector"
    ], bg_color="#EFF6FF", border_color="#93C5FD", header_color="#1D4ED8")

    draw_card(330, 380, 250, 170, "Similarity Retriever", "FAISS Search & Top-K", [
        "Top-K nearest neighbors",
        "Cosine similarity score [ -1, 1 ]",
        "Ranked candidate scoring",
        "Debug metrics & timing"
    ], bg_color="#FEF3C7", border_color="#FCD34D", header_color="#B45309")

    draw_card(620, 380, 250, 170, "Relevance Gatekeeper", "Strict Grounding Filter", [
        "Threshold check: score >= 0.38",
        "Pass: Send context to Gemini",
        "Reject: Immediate fallback",
        "Blocks out-of-scope queries"
    ], bg_color="#FFF1F2", border_color="#FECDD3", header_color="#BE123C")

    draw_card(910, 380, 250, 170, "Gemini LLM Generation", "Zero-Hallucination", [
        "Model: gemini-2.5-flash",
        "Temperature = 0.0 (deterministic)",
        "Strict grounded system prompt",
        "Never invents rules or policies"
    ], bg_color="#F0FDF4", border_color="#86EFAC", header_color="#15803D")

    # Arrows for Query Pipeline
    draw_arrow(292, 465, 328, 465)
    draw_arrow(582, 465, 618, 465)
    draw_arrow(872, 465, 908, 465)

    # OUTPUT SECTION
    draw.rounded_rectangle([(40, 580), (1160, 710)], radius=8, fill="#FFFFFF", outline="#1E3A8A", width=2)
    draw.text((60, 595), "3. VERIFIED OUTPUT & SOURCE ATTRIBUTION", fill="#1E3A8A", font=font_box_title)
    draw.text((60, 625), "• Grounded Answer: Concise, factual response synthesized exclusively from official policy provisions.", fill="#334155", font=font_text)
    draw.text((60, 648), "• Exact Fallback: 'I couldn't find enough information in the provided company documents to answer this question.'", fill="#B91C1C", font=font_text)
    draw.text((60, 671), "• Verified Attribution: Sources: 1. Leave_Policy.pdf — Page 1 | 2. Work_From_Home_Policy.pdf — Page 2 (No fabricated citations)", fill="#15803D", font=font_text)

    img.save(str(OUTPUT_PATH))
    print(f"Saved architecture diagram to: {OUTPUT_PATH}")


if __name__ == "__main__":
    create_diagram()
