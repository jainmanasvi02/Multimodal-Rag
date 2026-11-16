"""FAISS vector store over precomputed CLIP embeddings + scored retrieval."""
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .config import TOP_K
from .embeddings import CLIPEmbeddings, embed_text


def build_vector_store(docs: list[Document], embeddings: list[np.ndarray]) -> FAISS:
    arr = np.asarray(embeddings, dtype="float32")
    return FAISS.from_embeddings(
        text_embeddings=[(d.page_content, e.tolist()) for d, e in zip(docs, arr)],
        embedding=CLIPEmbeddings(),  # required by the API; stored vectors are precomputed
        metadatas=[d.metadata for d in docs],
    )


def retrieve(vector_store: FAISS, query: str, k: int = TOP_K):
    """Return [(Document, cosine_similarity)]. CLIP vectors are unit length and
    FAISS returns squared L2 distance, so cosine = 1 - dist / 2."""
    q = embed_text(query).astype("float32").tolist()
    results = vector_store.similarity_search_with_score_by_vector(q, k=k)
    return [(doc, 1 - float(dist) / 2) for doc, dist in results]
