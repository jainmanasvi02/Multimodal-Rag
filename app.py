"""Streamlit chat UI for the multimodal PDF RAG system."""
import base64
import os

import streamlit as st

from src.graph import build_graph, get_llm
from src.ingest import ingest_files
from src.retrieval import build_vector_store

st.set_page_config(page_title="Multimodal PDF RAG", page_icon="📄", layout="wide")

# API key: env var / .env (Codespaces secret) or Streamlit secrets (Streamlit Cloud)
if "GOOGLE_API_KEY" not in os.environ:
    try:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass

st.title("📄 Multimodal PDF Chat")
st.caption("CLIP + FAISS retrieval over PDF text and images, answered by Gemini via a LangGraph workflow.")

if "GOOGLE_API_KEY" not in os.environ:
    st.error("GOOGLE_API_KEY is not set. Add it to a .env file or your secrets and reload.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state["messages"] = []


def render_sources(retrieved, images):
    """Show retrieved items inside one expander (Streamlit can't nest expanders)."""
    with st.expander("Retrieved sources"):
        for doc, sim, used in retrieved:
            meta = doc.metadata
            label = (
                f"{meta['source']} · page {meta['page'] + 1} · "
                f"similarity {sim:.2f} · {'used' if used else 'filtered out'}"
            )
            if meta["type"] == "text":
                st.markdown(f"**Text** · {label}")
                st.write(doc.page_content)
            else:
                st.markdown(f"**Image** · {label}")
                b64 = images.get(meta["image_id"])
                if b64:
                    st.image(base64.b64decode(b64), width=350)
            st.divider()


# ---------------- sidebar: ingestion ----------------
with st.sidebar:
    st.header("Documents")
    files = st.file_uploader("PDF files", type="pdf", accept_multiple_files=True)
    if st.button("Build index", disabled=not files, type="primary"):
        with st.spinner("Reading PDFs and computing CLIP embeddings..."):
            docs, embs, images = ingest_files([(f.name, f.getvalue()) for f in files])
            if not docs:
                st.error("No text or images found in the uploaded PDFs.")
            else:
                store = build_vector_store(docs, embs)
                st.session_state["graph"] = build_graph(store, images, get_llm())
                st.session_state["images"] = images
                n_img = sum(d.metadata["type"] == "image" for d in docs)
                st.session_state["stats"] = (len(docs) - n_img, n_img)
                st.session_state["messages"] = []  # new documents -> fresh chat

    if "stats" in st.session_state:
        t, i = st.session_state["stats"]
        st.success(f"Indexed {t} text chunks and {i} images.")

    if st.button("Clear chat"):
        st.session_state["messages"] = []
        st.rerun()

# ---------------- chat history ----------------
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("searched"):
            st.caption(f"🔎 Searched for: {msg['searched']}")
        if msg.get("retrieved"):
            render_sources(msg["retrieved"], st.session_state.get("images", {}))

# ---------------- new question ----------------
query = st.chat_input("Ask a question about your document(s)")

if query:
    if "graph" not in st.session_state:
        st.warning("Upload a PDF in the sidebar and click 'Build index' first.")
    else:
        # earlier turns only (the new question isn't added to the history yet)
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state["messages"]
        ]
        st.session_state["messages"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            searched = None
            try:
                with st.spinner("Thinking..."):
                    result = st.session_state["graph"].invoke(
                        {"query": query, "history": history}
                    )
                answer, retrieved = result["answer"], result["retrieved"]
                sq = result.get("standalone_query")
                searched = sq if sq and sq != query else None
                st.write(answer)
                if searched:
                    st.caption(f"🔎 Searched for: {searched}")
                render_sources(retrieved, st.session_state["images"])
            except Exception as exc:
                answer, retrieved = f"Sorry, something went wrong: {exc}", []
                st.error(answer)

        st.session_state["messages"].append(
            {"role": "assistant", "content": answer, "retrieved": retrieved, "searched": searched}
        )