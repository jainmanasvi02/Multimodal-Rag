"""PDF ingestion: split text into chunks, extract images, embed everything with CLIP."""
import base64
import io

import fitz  # PyMuPDF
import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image

from .config import CHUNK_OVERLAP, CHUNK_SIZE, MAX_IMAGE_SIDE
from .embeddings import embed_image, embed_text


def ingest_pdf(name: str, data: bytes):
    """Return (docs, embeddings, image_store) for one PDF given as bytes."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    docs: list[Document] = []
    embeddings: list[np.ndarray] = []
    image_store: dict[str, str] = {}  # image_id -> base64 JPEG

    pdf = fitz.open(stream=data, filetype="pdf")
    try:
        for page_no, page in enumerate(pdf):
            # ---- text ----
            text = page.get_text()
            if text.strip():
                base = Document(
                    page_content=text,
                    metadata={"source": name, "page": page_no, "type": "text"},
                )
                for chunk in splitter.split_documents([base]):
                    embeddings.append(embed_text(chunk.page_content))
                    docs.append(chunk)

            # ---- images ----
            for img_no, img in enumerate(page.get_images(full=True)):
                try:
                    raw = pdf.extract_image(img[0])["image"]
                    pil = Image.open(io.BytesIO(raw)).convert("RGB")
                    pil.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))

                    image_id = f"{name}_page_{page_no}_img_{img_no}"
                    buf = io.BytesIO()
                    pil.save(buf, format="JPEG", quality=85)
                    image_store[image_id] = base64.b64encode(buf.getvalue()).decode()

                    embeddings.append(embed_image(pil))
                    docs.append(
                        Document(
                            page_content=f"[Image: {image_id}]",
                            metadata={
                                "source": name,
                                "page": page_no,
                                "type": "image",
                                "image_id": image_id,
                            },
                        )
                    )
                except Exception as exc:  # skip unreadable images, keep going
                    print(f"Skipping image {img_no} on page {page_no} of {name}: {exc}")
    finally:
        pdf.close()

    return docs, embeddings, image_store


def ingest_files(files: list[tuple[str, bytes]]):
    """Ingest several PDFs: files is a list of (filename, bytes)."""
    all_docs, all_embs, all_images = [], [], {}
    for name, data in files:
        docs, embs, images = ingest_pdf(name, data)
        all_docs += docs
        all_embs += embs
        all_images.update(images)
    return all_docs, all_embs, all_images
