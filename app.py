"""
HRQuery — AI-Powered HR & Company Policy Assistant
Streamlit Web Application
"""

import os
import sys
from pathlib import Path
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DEFAULT_TOP_K,
    DEFAULT_RELEVANCE_THRESHOLD,
    GEMINI_API_KEY,
    DEFAULT_GEMINI_MODEL,
    EMBEDDING_MODEL
)
from pipeline import HRQueryPipeline

# Streamlit Page Configuration
st.set_page_config(
    page_title="HRQuery — AI-Powered HR & Policy Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    /* Theme Accents */
    .main-header {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #1E3A8A;
        font-weight: 800;
        font-size: 2.2rem;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #4B5563;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .answer-box {
        background-color: #F8FAFC;
        border-left: 5px solid #2563EB;
        padding: 1.2rem;
        border-radius: 0.4rem;
        font-size: 1.02rem;
        line-height: 1.6;
        color: #1E293B;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .fallback-box {
        background-color: #FFFBEB;
        border-left: 5px solid #F59E0B;
        padding: 1.2rem;
        border-radius: 0.4rem;
        font-size: 1.02rem;
        color: #92400E;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .source-card {
        background: #F1F5F9;
        border: 1px solid #CBD5E1;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.92rem;
    }
    .metric-container {
        background: #EFF6FF;
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
        border: 1px solid #BFDBFE;
    }
    .chunk-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 0.8rem;
        margin-bottom: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Pipeline in Session State
@st.cache_resource
def get_pipeline(api_key: str, model_name: str):
    p = HRQueryPipeline(api_key=api_key)
    p.generator.model = model_name
    return p


# Session state for conversation history and active question
if "history" not in st.session_state:
    st.session_state.history = []
if "query_input" not in st.session_state:
    st.session_state.query_input = ""
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Ask Question"


# Sidebar Layout
with st.sidebar:
    st.image("https://img.icons8.com/color/96/briefcase--v1.png", width=64)
    st.title("HRQuery Control")
    st.caption("AI-Powered HR & Policy Knowledge Engine")

    # API Key Section
    st.subheader("🔑 Authentication")
    current_key = os.getenv("GEMINI_API_KEY", "") or GEMINI_API_KEY
    user_api_key = st.text_input(
        "Gemini API Key",
        value=current_key if current_key and current_key != "your_gemini_api_key_here" else "",
        type="password",
        help="Enter your Google Gemini API key. If omitted, pipeline runs in deterministic offline test mode."
    )
    if user_api_key:
        os.environ["GEMINI_API_KEY"] = user_api_key

    model_options = ["gemini-3.6-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    selected_model = st.selectbox(
        "Generation Model",
        options=model_options,
        index=0,
        help="Select the Gemini model for response generation (gemini-3.6-flash is recommended)."
    )

    pipeline = get_pipeline(user_api_key, selected_model)
    if pipeline.generator.model != selected_model:
        pipeline.generator.model = selected_model
    if pipeline.api_key != user_api_key:
        pipeline.set_api_key(user_api_key)


    if pipeline.generator.is_live:
        st.success("🟢 Connected to Google Gemini API", icon="✅")
    else:
        st.info("🟡 Running in Offline Fallback Mode", icon="ℹ️")

    st.markdown("---")

    # Knowledge Base Status Metrics
    st.subheader("📊 Knowledge Base Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Documents", pipeline.vector_store.total_documents)
    with col2:
        st.metric("Chunks", pipeline.vector_store.total_chunks)
    
    st.caption(f"**Index Type:** FAISS IndexFlatIP (Cosine)")
    st.caption(f"**Embedding Model:** `{EMBEDDING_MODEL}` (768-d)")
    st.caption(f"**LLM Model:** `{pipeline.generator.model}`")


    st.markdown("---")

    # Document Upload Section
    st.subheader("📂 Ingest Documents")
    uploaded_files = st.file_uploader(
        "Upload Policies (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Upload company HR policies, handbooks, or guideline documents."
    )

    if st.button("📥 Process Documents", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning("Please select at least one file to upload.")
        else:
            with st.spinner("Parsing, chunking, and embedding documents..."):
                results = pipeline.ingest_multiple_documents(uploaded_files)
                success_count = sum(1 for r in results if r["status"] == "success")
                st.success(f"Processed {len(results)} files ({success_count} newly indexed).")
                st.rerun()

    if st.button("🔄 Rebuild Knowledge Base", use_container_width=True, help="Clears vector store and re-indexes sample policies"):
        with st.spinner("Rebuilding knowledge base from NovaTech sample documents..."):
            pipeline.rebuild_knowledge_base(include_samples=True)
            st.success("Knowledge base rebuilt with 8 NovaTech sample policies!")
            st.rerun()

    st.markdown("---")

    # Retrieval Tuning
    st.subheader("⚙️ Retrieval Parameters")
    top_k = st.slider("Top-K Chunks", min_value=1, max_value=10, value=DEFAULT_TOP_K, step=1)
    relevance_threshold = st.slider(
        "Relevance Threshold (Cosine)",
        min_value=0.10,
        max_value=0.80,
        value=DEFAULT_RELEVANCE_THRESHOLD,
        step=0.02,
        help="Minimum cosine similarity required for retrieved chunks. Scores below this trigger the grounding fallback."
    )


# Main Content Layout
st.markdown('<div class="main-header">🏢 HRQuery: Policy Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Ask questions about NovaTech Solutions company policies. Answers are strictly grounded '
    'in retrieved official documents with verified source citations.'
    '</div>', 
    unsafe_allow_html=True
)

# If vector store is empty, prompt to seed sample docs
if pipeline.vector_store.total_chunks == 0:
    st.info(
        "💡 The knowledge base is currently empty. Click below to load the fictional NovaTech Solutions sample policies "
        "(Leave Policy, Employee Handbook, Code of Conduct, WFH Policy, Travel & Expense, etc.)."
    )
    if st.button("🚀 Load NovaTech Sample Documents Now", type="primary"):
        with st.spinner("Ingesting sample policies..."):
            pipeline.ingest_sample_documents()
            st.success("Sample policies successfully indexed!")
            st.rerun()

# Quick Demo Questions
st.markdown("**Sample Demo Queries:**")
demo_cols = st.columns(4)
demo_questions = [
    ("🏖️ Casual Leaves", "How many casual leaves are employees entitled to?"),
    ("🏡 Work From Home", "Can an employee work from home during a temporary health issue?"),
    ("⚖️ Code of Conduct", "What happens if an employee violates the company's code of conduct?"),
    ("✈️ Travel Expenses", "How are travel expenses reimbursed?"),
    ("🚪 Resignation Notice", "What is the resignation notice period?"),
    ("💰 Salary Query (Out of Scope)", "What is the salary of a software engineer with three years of experience?"),
    ("🍕 Cafeteria Menu (Out of Scope)", "What is the cafeteria menu?"),
]

def set_query(q_text):
    st.session_state.query_input = q_text

for i, (label, query_text) in enumerate(demo_questions):
    col = demo_cols[i % 4]
    if col.button(label, key=f"demo_btn_{i}", use_container_width=True):
        st.session_state.query_input = query_text

# Query Input
user_query = st.text_input(
    "Ask a policy question:",
    value=st.session_state.query_input,
    placeholder="e.g., How many casual leaves can I take at once?",
    key="main_query_input"
)

col_ask, col_clear = st.columns([1, 5])
with col_ask:
    ask_button = st.button("🔍 Ask Question", type="primary", use_container_width=True)

# Process Question
if ask_button and user_query.strip():
    with st.spinner("Searching company documents and generating answer..."):
        response = pipeline.answer_question(
            question=user_query.strip(),
            top_k=top_k,
            threshold=relevance_threshold
        )
        
        # Save to session history
        st.session_state.history.insert(0, response)

# Display Current or Most Recent Result
if st.session_state.history:
    latest = st.session_state.history[0]

    st.markdown("### 💬 Answer")
    if latest["is_fallback"]:
        st.markdown(
            f'<div class="fallback-box">⚠️ <b>Grounding Fallback:</b><br>{latest["answer"]}</div>',
            unsafe_allow_html=True
        )
        st.caption("ℹ️ This question could not be answered from the official policies. The strict grounding gatekeeper blocked hallucination.")
    else:
        st.markdown(
            f'<div class="answer-box">{latest["answer"]}</div>',
            unsafe_allow_html=True
        )

        # Sources Section
        st.markdown("### 📚 Source Attribution")
        if latest["sources"]:
            cols = st.columns(min(len(latest["sources"]), 3))
            for idx, (doc, page) in enumerate(latest["sources"]):
                col = cols[idx % len(cols)]
                page_label = f"Page {page}" if (page and str(page) != "0") else "Page unavailable"
                with col:
                    st.markdown(
                        f"""
                        <div class="source-card">
                            📄 <b>{doc}</b><br>
                            <span style="color: #64748B;">📍 {page_label}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.info("No sources cited.")

    # Debug Inspection Section
    with st.expander("🔎 Retrieved Context & Retrieval Quality (Debug Inspector)", expanded=False):
        st.markdown(f"**Retrieval Diagnostics:** `{latest['debug_summary']}`")
        st.markdown(f"- **Top Score:** `{latest['top_score']:.4f}`")
        st.markdown(f"- **Threshold:** `{latest['threshold']:.2f}`")
        st.markdown(f"- **Relevance Gate:** `{'PASSED' if latest['is_relevant'] else 'FALLBACK (Rejected)'}`")
        st.markdown(f"- **Retrieval Latency:** `{latest['latency_ms']} ms`")

        show_all = st.checkbox("Show all raw candidate chunks", value=False)
        display_chunks = latest["raw_candidates"] if show_all else latest["chunks_used"]

        if not display_chunks:
            st.write("No chunks passed the relevance filter.")
        else:
            for rank, chunk in enumerate(display_chunks, start=1):
                page_str = f"Page {chunk.get('page_number', 'N/A')}"
                score = chunk.get("similarity_score", 0.0)
                st.markdown(
                    f"""
                    <div class="chunk-card">
                        <b>Rank {rank}</b> | <b>{chunk.get('document_name')}</b> ({page_str}) | 
                        <b>Similarity Score:</b> <code>{score:.4f}</code> | 
                        <b>Chunk ID:</b> <code>{chunk.get('chunk_id')}</code>
                        <hr style="margin: 0.4rem 0;">
                        <span style="color: #334155; font-size: 0.93rem;">{chunk.get('text')}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# Conversation History Accordion
if len(st.session_state.history) > 1:
    st.markdown("---")
    st.markdown("### 🕒 Recent Questions")
    for past in st.session_state.history[1:6]:
        with st.expander(f"Q: {past['question']}", expanded=False):
            st.write(past["answer"])
            if past["sources_formatted"]:
                st.markdown(past["sources_formatted"])
            st.caption(f"Score: {past['top_score']:.4f} | Relevant: {past['is_relevant']}")
