# Multimodal PDF RAG

Ask questions about PDFs that contain both **text and images** (charts, diagrams).
CLIP embeds text chunks and images, FAISS retrieves the most relevant items, and a
Gemini model answers from the retrieved text and images. A **LangGraph** workflow
rewrites follow-up questions, retrieves, and refuses to answer when nothing relevant
is found. The project includes a Streamlit chat UI and an evaluation harness with
sample PDFs.

## Features

- **Multimodal retrieval:** PDF text chunks and embedded images are indexed together, and charts and diagrams are sent to Gemini as images.
- **Separate text and image search:** CLIP scores images much lower than text, so the two are searched separately. Searching them together never returned an image (0/7 image questions); separating them raised that to 100%.
- **Multiple PDFs:** upload several files at once; sources show the file name and page.
- **Follow-up questions:** a LangGraph step rewrites vague follow-ups ("what about Q3?") into standalone questions using the chat history.
- **Relevance gate:** if no retrieved item passes its similarity threshold, the graph answers "not found" without calling the LLM. The prompt also tells Gemini to refuse when the context lacks the answer.
- **Chunk labels and de-duplication:** each text chunk is embedded with a "document name, page N" prefix, and identical repeated chunks (boilerplate) are indexed once.
- **Evaluation harness:** sample PDFs, labeled questions, retrieval metrics and an optional answer check.

## Architecture

```
PDF -> text chunks + images -> CLIP embeddings -> FAISS index
        (chunks labeled with document and page, duplicates removed)

question (+ chat history)
   -> rewrite as standalone question (only if there is history)
   -> embed with CLIP -> search text chunks (top 5) and images (top 2) separately
   -> relevance gate (similarity thresholds for text and images)
        pass -> Gemini (text + images) -> answer
        fail -> "not found in the document"
```

## Project structure

```
app.py                     Streamlit chat UI
src/config.py              settings and thresholds
src/embeddings.py          CLIP text/image embeddings
src/ingest.py              PDF parsing, chunking, image extraction, de-duplication
src/retrieval.py           FAISS store and scored retrieval (text and images searched separately)
src/graph.py               LangGraph workflow and prompt building
eval/run_eval.py           evaluation script
eval/questions.json        27 questions over 3 sample PDFs
eval/questions_large.json  84 questions over 15 PDFs
eval/make_sample_pdfs.py   generates the 3 sample PDFs
eval/make_large_corpus.py  generates the 15-PDF corpus
data/samples/              3 sample PDFs
data/corpus/               15 PDFs (larger corpus)
```

## Run it

```bash
pip install -r requirements.txt
echo "GOOGLE_API_KEY=your_key_here" > .env
streamlit run app.py
```

Get a free Gemini key at https://aistudio.google.com. Upload PDFs in the sidebar,
click **Build index**, then ask questions in the chat box.

### GitHub Codespaces

Open the repo in a Codespace; dependencies install automatically. Add `GOOGLE_API_KEY`
as a Codespaces secret (or in `.env`), run `streamlit run app.py`, and open the forwarded port.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_MODEL` | `gemini-3.8-flash-lite` | Gemini model name (model names change often) |
| `TEXT_MIN_SIM` | `0.55` | Minimum CLIP similarity for a text chunk to count as relevant |
| `IMAGE_MIN_SIM` | `0.20` | Minimum CLIP similarity for an image to count as relevant |

Retrieval depth is set in `src/config.py` (`TOP_K` for text, `TOP_K_IMAGES` for images).

## Evaluation

All numbers come from `eval/run_eval.py` on synthetic PDFs generated for this project
(text, bar/pie/line charts and labeled diagrams, with known answers).

```bash
# retrieval metrics only (CLIP + FAISS, no API calls)
python eval/run_eval.py --pdf-dir data/corpus --questions eval/questions_large.json --out-dir eval/large_clip

# also check final answers with Gemini (uses API quota)
python eval/run_eval.py --with-llm --sleep 3
```

Metrics: **Hit@1 / Hit@5 / MRR** (the expected page, and for text questions the chunk
containing the answer, is among the top results), plus the share of off-topic
questions blocked by the relevance gate.

### Small corpus (3 PDFs, 27 questions)

| Step | Text Hit@1 | Image Hit@1 |
|---|---|---|
| CLIP, text and images searched together | 46% | 0% |
| Text and images searched separately | 46% | 100% |

With Gemini on top of the separated search, all 27 answers were judged correct,
including 7 off-topic questions answered with "not found".

### Larger corpus (15 PDFs, 51 text / 17 image / 16 off-topic questions)

The corpus deliberately contains near-identical documents (four company annual
reports, similar appliance manuals, several studies with the same layout).

| Step | Text Hit@1 | Text Hit@5 | Text MRR | Image Hit@1 |
|---|---|---|---|---|
| CLIP only | 33% | 51% | 0.39 | 100% |
| + de-duplicate repeated chunks | 33% | 61% | 0.43 | 100% |
| + document/page label on chunks | 22% | 76% | 0.42 | 100% |

Findings:

- The label raised recall (Hit@5) but reshuffled the ranking: 34 of 51 questions changed rank, 11 dropped out of first place, and 5 gained it. The app sends the top 5 text chunks to Gemini, so Hit@5 is the metric that matters most here.
- The relevance gate blocks **0%** of off-topic questions: CLIP's similarity scores for unrelated text overlap with the scores of correct matches. Refusal currently depends on the prompt instruction to Gemini. (The image similarity scores do separate cleanly: correct images score about 0.29 to 0.45 and the best image for unrelated questions about 0.23 or less.)
- The Gemini answer check was only run on the small corpus because of API quota.

## Limitations

- CLIP is weak at matching dense text, and its text encoder reads only the first 77 tokens of a chunk.
- The text similarity thresholds are not usable as a relevance gate with CLIP (see above).
- Only embedded raster images are indexed, not vector-drawn charts.
- The evaluation set is small and synthetic, so the numbers show relative changes more than real-world accuracy.
- Identical chunks are indexed once, so a repeated chunk is attributed to the first file it appeared in.

## Future work

- Add a dedicated text embedding model for text chunks (keeping CLIP for images), with separate thresholds so the relevance gate works. *In progress.*
- Run the Gemini answer check on the larger corpus.
- Test on real, messy documents (scans, tables).

## Acknowledgements

Started from the multimodal RAG lesson in Krish Naik's LangGraph course, then extended
with the separate image search, follow-up question rewriting, relevance gate, chat UI,
de-duplication, chunk labels and the evaluation harness.

## License

MIT

**Note on dependencies:** PDF parsing uses [PyMuPDF](https://github.com/pymupdf/PyMuPDF),
which is licensed under AGPL-3.0 (a commercial license is also available from Artifex).
If you plan to deploy this project as a public service or distribute it commercially,
please review the PyMuPDF license terms first. Other dependencies have their own licenses.

## Author

Manasvi Jain