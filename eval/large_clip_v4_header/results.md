# Evaluation results

Embedding model: `openai/clip-vit-base-patch32` (CLIP, text and images) | k = 5 | thresholds: text >= 0.55, image >= 0.2
Corpus: 15 PDFs, 63 text chunks, 14 images | Questions: 51 text, 17 image, 16 off-topic

## Retrieval (answerable questions)

| Metric | Text questions | Image questions | All |
|---|---|---|---|
| Hit@1 | 22% | 100% | 41% |
| Hit@5 | 76% | 100% | 82% |
| MRR | 0.42 | 1.00 | 0.56 |
| Expected source kept after relevance gate | 76% | 100% | 82% |
| Relevance gate lets the question through | 100% | 100% | 100% |

## Refusal (off-topic questions)

| Type | Questions | Blocked by gate (no LLM call) |
|---|---|---|
| Easy (unrelated topics) | 8 | 0% |
| Hard (same domain, not in the documents) | 8 | 0% |

## Retrieval misses

- `t05` What revenue target has Acme set for 2026? -> expected acme_annual_report.pdf p.4 (text); top result: acme_annual_report.pdf p.1 (text, sim 0.8032)
- `t08` What does error E7 mean on the coffee machine? -> expected brewmaster_manual.pdf p.3 (text); top result: employee_handbook.pdf p.1 (text, sim 0.6918)
- `t09` How often should the coffee machine be descaled? -> expected brewmaster_manual.pdf p.3 (text); top result: employee_handbook.pdf p.1 (text, sim 0.7732)
- `t10` How many buoys were used in the temperature study? -> expected climate_study_summary.pdf p.1 (text); top result: climate_study_summary.pdf p.3 (text, sim 0.7641)
- `t21` What revenue target has Cedarline Foods set for 2026? -> expected cedarline_annual_report.pdf p.4 (text); top result: cedarline_annual_report.pdf p.1 (text, sim 0.8408)
- `t22` What dividend per share did the Cedarline Foods board propose? -> expected cedarline_annual_report.pdf p.4 (text); top result: cedarline_annual_report.pdf p.1 (text, sim 0.8398)
- `t30` What is the maximum capacity of the KettleMax 1.7? -> expected kettlemax_manual.pdf p.1 (text); top result: employee_handbook.pdf p.1 (text, sim 0.7101)
- `t31` How often should the KettleMax be descaled? -> expected kettlemax_manual.pdf p.3 (text); top result: employee_handbook.pdf p.1 (text, sim 0.7494)
- `t37` Who funded the urban air quality study? -> expected urban_air_quality_study.pdf p.3 (text); top result: urban_air_quality_study.pdf p.1 (text, sim 0.8406)
- `t38` What statistical model was used in the urban air quality study? -> expected urban_air_quality_study.pdf p.3 (text); top result: urban_air_quality_study.pdf p.1 (text, sim 0.8605)
- `t39` How many plots were used in the soil moisture trial? -> expected soil_moisture_trial.pdf p.1 (text); top result: employee_handbook.pdf p.1 (text, sim 0.7343)
- `t46` What is the daily meal allowance in the travel policy? -> expected travel_expense_policy.pdf p.1 (text); top result: employee_handbook.pdf p.1 (text, sim 0.7816)

## Suggested thresholds

`TEXT_MIN_SIM=0.775` and `IMAGE_MIN_SIM=0.26` (on this question set: expected source kept 41%, easy off-topic blocked 100%; 5 threshold pairs scored equally well). Tuned on the same questions it is reported on, so treat it as a starting point.
