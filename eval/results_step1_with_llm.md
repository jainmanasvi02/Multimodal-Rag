# Evaluation results

Embedding model: `openai/clip-vit-base-patch32` (CLIP, text and images) | k = 5 | thresholds: text >= 0.55, image >= 0.2
Corpus: 3 PDFs, 10 text chunks, 4 images | Questions: 13 text, 7 image, 7 off-topic

## Retrieval (answerable questions)

| Metric | Text questions | Image questions | All |
|---|---|---|---|
| Hit@1 | 46% | 100% | 65% |
| Hit@5 | 100% | 100% | 100% |
| MRR | 0.67 | 1.00 | 0.78 |
| Expected source kept after relevance gate | 100% | 100% | 100% |
| Relevance gate lets the question through | 100% | 100% | 100% |

## Refusal (off-topic questions)

| Type | Questions | Blocked by gate (no LLM call) |
|---|---|---|
| Easy (unrelated topics) | 4 | 0% |
| Hard (same domain, not in the documents) | 3 | 0% |

## Answer correctness (with Gemini)

| Question type | Questions | Correct |
|---|---|---|
| Text | 13 | 100% |
| Image | 7 | 100% |
| Off-topic, easy (should refuse) | 4 | 100% |
| Off-topic, hard (should refuse) | 3 | 100% |

Requests that errored (counted as wrong): 0

## Suggested thresholds

`TEXT_MIN_SIM=0.75` and `IMAGE_MIN_SIM=0.26` (on this question set: expected source kept 65%, easy off-topic blocked 100%; 5 threshold pairs scored equally well). Tuned on the same questions it is reported on, so treat it as a starting point.
