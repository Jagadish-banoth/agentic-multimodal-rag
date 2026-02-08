import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import streamlit as st
from ingestion.text_ingest import ingest_text
from ingestion.image_ingest import ingest_image
from ingestion.audio_ingest import ingest_audio
from ingestion.video_ingest import ingest_video
from orchestrator.execution_engine import ExecutionEngine
from ui._index_utils import run_embed_and_index, deduplicate_raw_files, cleanup_temp_files

# Create a singleton ExecutionEngine instance
import yaml
from pathlib import Path
@st.cache_resource(show_spinner=False)
def get_engine():
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return ExecutionEngine(config)

engine = get_engine()

# --- UI CONFIG ---
st.set_page_config(page_title="Agentic Multimodal RAG", layout="wide", initial_sidebar_state="expanded")

# --- LIGHT/DARK MODE ---
mode = st.sidebar.radio("Theme", ["Light", "Dark"])
if mode == "Dark":
    st.markdown(
        """
        <style>
        body { background-color: #181818; color: #f1f1f1; }
        .stApp { background-color: #181818; }
        </style>
        """, unsafe_allow_html=True
    )

# --- HEADER ---
st.title("🤖 Agentic Multimodal RAG Chat")
st.caption("Upload documents (text, image, audio, video) and ask anything!")

# --- FILE UPLOAD ---

uploaded_file = st.file_uploader(
    "Upload a document, image, audio, or video",
    type=["txt", "pdf", "png", "jpg", "jpeg", "mp3", "wav", "mp4", "avi", "mov"],
    accept_multiple_files=False
)

# --- INDEX REBUILD BUTTON ---
st.sidebar.markdown("---")
if st.sidebar.button("Rebuild Index", help="Manually rebuild FAISS/BM25 indices after ingestion"):
    with st.spinner("Rebuilding index (this may take a while)..."):
        out, err, code = run_embed_and_index()
        if code == 0:
            st.sidebar.success("Index rebuilt successfully!")
        else:
            st.sidebar.error(f"Index rebuild failed: {err}")

# --- INGESTION ---

doc_id = None
if uploaded_file:
    filetype = uploaded_file.type
    if "text" in filetype or uploaded_file.name.endswith((".txt", ".pdf")):
        doc_id = ingest_text(uploaded_file)
        st.success("Text document ingested.")
    elif "image" in filetype or uploaded_file.name.endswith((".png", ".jpg", ".jpeg")):
        doc_id = ingest_image(uploaded_file)
        st.success("Image ingested.")
    elif "audio" in filetype or uploaded_file.name.endswith((".mp3", ".wav")):
        doc_id = ingest_audio(uploaded_file)
        st.success("Audio ingested.")
    elif "video" in filetype or uploaded_file.name.endswith((".mp4", ".avi", ".mov")):
        doc_id = ingest_video(uploaded_file)
        st.success("Video ingested.")
    else:
        st.error("Unsupported file type.")
        doc_id = None

    # --- Deduplicate and cleanup temp files ---
    raw_dirs = [
        Path("data/raw/text"),
        Path("data/raw/audio"),
        Path("data/raw/video"),
    ]
    for raw_dir in raw_dirs:
        if raw_dir.exists():
            deduplicate_raw_files(raw_dir)
            cleanup_temp_files(raw_dir, keep_latest=1)

    # --- Auto-run embedding and indexing after upload ---
    with st.spinner("Embedding and indexing uploaded content (please wait)..."):
        out, err, code = run_embed_and_index()
        if code == 0:
            st.success("Index updated!")
        else:
            st.error(f"Indexing failed: {err}")

# --- CHAT INTERFACE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask anything about your uploaded content:")


if st.button("Send") and query:
    with st.spinner("Processing..."):
        # Run the full agentic pipeline using ExecutionEngine
        # If doc_id is needed, pass as part of query or context (customize as needed)
        response = engine.run(query)
        answer = response.get("answer", "No answer")
        sources = response.get("sources", [])
        confidence = response.get("confidence", 0.0)
        st.session_state.chat_history.append(
            {"query": query, "answer": answer, "sources": sources, "confidence": confidence}
        )

# --- DISPLAY CHAT HISTORY ---
for chat in st.session_state.chat_history[::-1]:
    st.markdown(f"**You:** {chat['query']}")
    st.markdown(f"**Agent:** {chat['answer']}")
    st.markdown(f"_Sources:_ {chat['sources']} | _Confidence:_ {chat['confidence']:.2f}")
    st.markdown("---")