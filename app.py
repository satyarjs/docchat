"""Streamlit web UI for DocChat.

Run with:  streamlit run app.py

Important Streamlit idea: this WHOLE script re-runs from top to bottom on every
interaction (every click, every message). Two tools make that workable:
  * @st.cache_resource -> create the DocChat object (and its 90 MB embedding
    model) ONCE per server process instead of on every rerun.
  * st.session_state   -> a dict that survives reruns; we keep chat history in it.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from docchat import DocChat, get_settings
from docchat.llm import LLMError

UPLOAD_DIR = Path("data/uploads")
SAMPLE_FILE = Path("data/sample.txt")

st.set_page_config(page_title="DocChat", page_icon="📄")


@st.cache_resource
def get_app() -> DocChat:
    """Create DocChat once and load any index already saved on disk."""
    app = DocChat(get_settings())
    app.load_index()
    return app


def render_sources(sources: list[dict], used_llm: bool) -> None:
    """Show where an answer came from."""
    if sources:
        with st.expander("Sources"):
            for s in sources:
                st.markdown(
                    f"**[{s['n']}] {s['source']} — page {s['page']} — score {s['score']:.2f}**"
                )
                st.caption(s["text"])
    elif not used_llm:
        st.caption("No relevant passages found — the LLM was not called.")


try:
    app = get_app()
except (ValueError, FileNotFoundError) as exc:
    st.error(f"Could not start DocChat: {exc}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content", "sources", "used_llm"}

# ---------------- Sidebar: documents and index ----------------
with st.sidebar:
    st.header("📚 Documents")
    uploaded = st.file_uploader(
        "Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
    )
    include_sample = st.checkbox("Include sample document", value=True)

    if st.button("Build index", type="primary"):
        paths: list[Path] = []
        if uploaded:
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            for file in uploaded:
                target = UPLOAD_DIR / file.name
                target.write_bytes(file.getbuffer())
                paths.append(target)
        if include_sample and SAMPLE_FILE.exists():
            paths.append(SAMPLE_FILE)

        if not paths:
            st.warning("Upload at least one file or tick 'Include sample document'.")
        else:
            try:
                with st.spinner("Reading, chunking and embedding... (first run downloads the model)"):
                    report = app.ingest(paths)
                st.success(
                    f"Indexed {report.files} file(s), {report.pages} page(s), "
                    f"{report.chunks} chunk(s)."
                )
                st.session_state.messages = []  # old answers may refer to old docs
            except (ValueError, FileNotFoundError) as exc:
                st.error(str(exc))

    st.divider()
    if app.is_ready:
        st.write(f"✅ Index ready: **{len(app.store)}** chunks")
    else:
        st.write("No index yet.")

    with st.expander("Settings (edit .env to change)"):
        s = app.settings
        st.write(f"LLM model: `{s.llm_model}`")
        st.write(f"LLM URL: `{s.llm_base_url}`")
        st.write(f"Embedding model: `{s.embedding_model}`")
        st.write(f"Chunk size / overlap: `{s.chunk_size}` / `{s.chunk_overlap}`")
        st.write(f"Top-k: `{s.top_k}` · Min score: `{s.min_score}`")

# ---------------- Main: chat ----------------
st.title("📄 DocChat")
st.caption("Ask questions about your own documents. Answers cite the page they came from.")

if not app.is_ready:
    st.info("Build the index from the sidebar first (the sample document works out of the box).")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg["sources"], msg["used_llm"])

question = st.chat_input("Ask about your documents…", disabled=not app.is_ready)
if question:
    st.session_state.messages.append(
        {"role": "user", "content": question, "sources": [], "used_llm": False}
    )
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching and thinking..."):
                answer = app.ask(question)
            sources = [
                {
                    "n": i,
                    "source": r.chunk.source,
                    "page": r.chunk.page_number,
                    "score": r.score,
                    "text": r.chunk.text,
                }
                for i, r in enumerate(answer.sources, start=1)
            ]
            st.markdown(answer.text)
            render_sources(sources, answer.used_llm)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer.text,
                    "sources": sources,
                    "used_llm": answer.used_llm,
                }
            )
        except (LLMError, RuntimeError, ValueError) as exc:
            st.error(str(exc))
