"""Central configuration. Override via environment variables / .env."""
import os

from dotenv import load_dotenv

load_dotenv()

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

TOP_K = 5

# Minimum cosine similarity for a retrieved item to count as "relevant".
# CLIP text-to-text scores run higher than text-to-image scores, so they differ.
TEXT_MIN_SIM = float(os.getenv("TEXT_MIN_SIM", "0.55"))
IMAGE_MIN_SIM = float(os.getenv("IMAGE_MIN_SIM", "0.20"))

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MAX_IMAGE_SIDE = 1024  # images are downscaled before being sent to the LLM
TOP_K_IMAGES = 2