# Multimodal PDF RAG

Ask questions about PDFs that contain both **text and images** (charts, diagrams).
CLIP embeds text chunks and images into one vector space, FAISS retrieves the most
relevant items, and a Gemini model answers from the retrieved text and images.
A **LangGraph** workflow gates the answer: if nothing relevant is retrieved, the
system says so instead of guessing.

## Architecture

```
PDF -> text chunks + images -> CLIP embeddings -> FAISS index
                                                     |
question -> CLIP embedding -> retrieve top-k -> relevance filter
                                                     |
                      relevant? --yes--> Gemini (text + images) --> answer
                              \--no---> "not found in the document"
```

## Project structure

```
app.py              Streamlit UI
src/config.py       settings and thresholds
src/embeddings.py   CLIP text/image embeddings
src/ingest.py       PDF parsing, chunking, image extraction
src/retrieval.py    FAISS store and scored retrieval
src/graph.py        LangGraph workflow and prompt building
```

## Run it

```bash
pip install -r requirements.txt
echo "GOOGLE_API_KEY=your_key_here" > .env
streamlit run app.py
```

Get a free Gemini key at https://aistudio.google.com. Upload PDFs in the sidebar,
click **Build index**, then ask questions.

### GitHub Codespaces
Open the repo in a Codespace; dependencies install automatically. Add `GOOGLE_API_KEY`
as a Codespaces secret (or in `.env`), run `streamlit run app.py`, and open the forwarded port.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_MODEL` | `gemini-3.8-flash-lite` | Gemini model name (they change often) |
| `TEXT_MIN_SIM` | `0.55` | Min CLIP similarity for text to count as relevant |
| `IMAGE_MIN_SIM` | `0.20` | Min CLIP similarity for images |

## Limitations / future work

- CLIP is weak on dense text; a dedicated text embedding model for text chunks would help.
- Only embedded raster images are indexed, not vector-drawn charts.
- Similarity thresholds are heuristic; an evaluation set would let them be tuned properly.

## Author
Manasvi Jain