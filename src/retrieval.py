"""FAISS vector store over precomputed CLIP embeddings + scored retrieval."""
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .config import TOP_K, TOP_K_IMAGES
from .embeddings import CLIPEmbeddings, embed_text


def build_vector_store(docs: list[Document], embeddings: list[np.ndarray]) -> FAISS:
    arr = np.asarray(embeddings, dtype="float32")
    return FAISS.from_embeddings(
        text_embeddings=[(d.page_content, e.tolist()) for d, e in zip(docs, arr)],
        embedding=CLIPEmbeddings(),  # required by the API; stored vectors are precomputed
        metadatas=[d.metadata for d in docs],
    )


def retrieve(vector_store: FAISS, query: str, k: int = TOP_K, k_images: int = TOP_K_IMAGES):
    """Search text and images separately (their CLIP score ranges differ)."""
    q = embed_text(query).astype("float32").tolist()
    n = vector_store.index.ntotal
    out = []
    for kind, kk in (("text", k), ("image", k_images)):
        results = vector_store.similarity_search_with_score_by_vector(
            q, k=kk, filter={"type": kind}, fetch_k=n
        )
        out += [(doc, 1 - float(dist) / 2) for doc, dist in results]
    return out
