"""CLIP embeddings for both text and images (shared vector space)."""
from functools import lru_cache

import numpy as np
import torch
from langchain_core.embeddings import Embeddings
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .config import CLIP_MODEL_NAME


@lru_cache(maxsize=1)
def _load():
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model.eval()
    return model, processor


def _to_tensor(out):
    """Newer transformers versions may return an output object instead of a tensor."""
    return out if isinstance(out, torch.Tensor) else out.pooler_output


def embed_image(image: Image.Image) -> np.ndarray:
    model, processor = _load()
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        feats = _to_tensor(model.get_image_features(**inputs))
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.squeeze().numpy()


def embed_text(text: str) -> np.ndarray:
    model, processor = _load()
    inputs = processor(
        text=text, return_tensors="pt", padding=True, truncation=True, max_length=77
    )
    with torch.no_grad():
        feats = _to_tensor(model.get_text_features(**inputs))
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.squeeze().numpy()


class CLIPEmbeddings(Embeddings):
    """LangChain Embeddings wrapper so FAISS accepts our CLIP model."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [embed_text(t).tolist() for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return embed_text(text).tolist()
