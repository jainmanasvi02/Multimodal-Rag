# Evaluation results

Embedding model: `openai/clip-vit-base-patch32` (CLIP, text and images) | k = 5 | thresholds: text >= 0.55, image >= 0.2
Corpus: 3 PDFs, 10 text chunks, 4 images | Questions: 13 text, 7 image, 7 off-topic

## Retrieval (answerable questions)

| Metric | Text questions | Image questions | All |
|---|---|---|---|
| Hit@1 | 46% | 0% | 30% |
| Hit@5 | 100% | 0% | 65% |
| MRR | 0.67 | 0.00 | 0.43 |
| Expected source kept after relevance gate | 100% | 0% | 65% |
| Relevance gate lets the question through | 100% | 100% | 100% |

## Refusal (off-topic questions)

| Type | Questions | Blocked by gate (no LLM call) |
|---|---|---|
| Easy (unrelated topics) | 4 | 0% |
| Hard (same domain, not in the documents) | 3 | 0% |

## Retrieval misses

- `i01` According to the pie chart, which region contributes the largest share of revenue? -> expected acme_annual_report.pdf p.3 (image); top result: acme_annual_report.pdf p.2 (text, sim 0.7306)
- `i02` What percentage of revenue comes from Asia in the revenue by region chart? -> expected acme_annual_report.pdf p.3 (image); top result: acme_annual_report.pdf p.2 (text, sim 0.7689)
- `i03` Which quarter has the tallest bar in the quarterly revenue chart? -> expected acme_annual_report.pdf p.2 (image); top result: acme_annual_report.pdf p.2 (text, sim 0.8162)
- `i04` In the coffee machine parts diagram, which component sits directly below the brew head? -> expected brewmaster_manual.pdf p.2 (image); top result: brewmaster_manual.pdf p.2 (text, sim 0.7657)
- `i05` Which components are labeled in the coffee machine parts diagram? -> expected brewmaster_manual.pdf p.2 (image); top result: brewmaster_manual.pdf p.2 (text, sim 0.731)
- `i06` In which year does the sea surface temperature chart reach its peak? -> expected climate_study_summary.pdf p.2 (image); top result: climate_study_summary.pdf p.3 (text, sim 0.8119)
- `i07` What does the vertical axis of the temperature chart measure? -> expected climate_study_summary.pdf p.2 (image); top result: climate_study_summary.pdf p.3 (text, sim 0.7824)

## Suggested thresholds

`TEXT_MIN_SIM=0.75` and `IMAGE_MIN_SIM=0.22` (on this question set: expected source kept 30%, easy off-topic blocked 100%; 25 threshold pairs scored equally well). Tuned on the same questions it is reported on, so treat it as a starting point.
