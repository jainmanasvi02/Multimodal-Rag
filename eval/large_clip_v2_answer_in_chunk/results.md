# Evaluation results

Embedding model: `openai/clip-vit-base-patch32` (CLIP, text and images) | k = 5 | thresholds: text >= 0.55, image >= 0.2
Corpus: 15 PDFs, 70 text chunks, 14 images | Questions: 51 text, 17 image, 16 off-topic

## Retrieval (answerable questions)

| Metric | Text questions | Image questions | All |
|---|---|---|---|
| Hit@1 | 33% | 100% | 50% |
| Hit@5 | 51% | 100% | 63% |
| MRR | 0.39 | 1.00 | 0.54 |
| Expected source kept after relevance gate | 51% | 100% | 63% |
| Relevance gate lets the question through | 100% | 100% | 100% |

## Refusal (off-topic questions)

| Type | Questions | Blocked by gate (no LLM call) |
|---|---|---|
| Easy (unrelated topics) | 8 | 0% |
| Hard (same domain, not in the documents) | 8 | 0% |

## Retrieval misses

- `t04` What was Acme's annual customer churn rate? -> expected acme_annual_report.pdf p.3 (text); top result: acme_annual_report.pdf p.1 (text, sim 0.7519)
- `t05` What revenue target has Acme set for 2026? -> expected acme_annual_report.pdf p.4 (text); top result: acme_annual_report.pdf p.1 (text, sim 0.8153)
- `t06` What is the capacity of the BrewMaster water tank? -> expected brewmaster_manual.pdf p.1 (text); top result: borealis_annual_report.pdf p.4 (text, sim 0.7697)
- `t07` How long is the BrewMaster 3000 warranty? -> expected brewmaster_manual.pdf p.1 (text); top result: airpure_200_manual.pdf p.3 (text, sim 0.7733)
- `t09` How often should the coffee machine be descaled? -> expected brewmaster_manual.pdf p.3 (text); top result: borealis_annual_report.pdf p.4 (text, sim 0.8009)
- `t10` How many buoys were used in the temperature study? -> expected climate_study_summary.pdf p.1 (text); top result: climate_study_summary.pdf p.3 (text, sim 0.8261)
- `t11` By how much did mean sea surface temperature rise between 2000 and 2024? -> expected climate_study_summary.pdf p.1 (text); top result: climate_study_summary.pdf p.3 (text, sim 0.8354)
- `t16` Which product drove Borealis Systems' growth in Q2? -> expected borealis_annual_report.pdf p.2 (text); top result: borealis_annual_report.pdf p.1 (text, sim 0.8283)
- `t17` What operating margin did Borealis Systems report for the year? -> expected borealis_annual_report.pdf p.2 (text); top result: borealis_annual_report.pdf p.1 (text, sim 0.8223)
- `t21` What revenue target has Cedarline Foods set for 2026? -> expected cedarline_annual_report.pdf p.4 (text); top result: cedarline_annual_report.pdf p.1 (text, sim 0.8616)
- `t22` What dividend per share did the Cedarline Foods board propose? -> expected cedarline_annual_report.pdf p.4 (text); top result: cedarline_annual_report.pdf p.1 (text, sim 0.856)
- `t25` Where is Delphi Logistics headquartered? -> expected delphi_annual_report.pdf p.1 (text); top result: urban_air_quality_study.pdf p.1 (text, sim 0.8122)
- `t28` What does code F5 mean on the AirPure 200? -> expected airpure_200_manual.pdf p.3 (text); top result: airpure_200_manual.pdf p.3 (text, sim 0.7375)
- `t30` What is the maximum capacity of the KettleMax 1.7? -> expected kettlemax_manual.pdf p.1 (text); top result: borealis_annual_report.pdf p.4 (text, sim 0.7875)
- `t31` How often should the KettleMax be descaled? -> expected kettlemax_manual.pdf p.3 (text); top result: borealis_annual_report.pdf p.4 (text, sim 0.8282)
- `t32` What does it mean when the kettle light flashes three times? -> expected kettlemax_manual.pdf p.3 (text); top result: borealis_annual_report.pdf p.4 (text, sim 0.7722)
- `t37` Who funded the urban air quality study? -> expected urban_air_quality_study.pdf p.3 (text); top result: urban_air_quality_study.pdf p.1 (text, sim 0.8109)
- `t38` What statistical model was used in the urban air quality study? -> expected urban_air_quality_study.pdf p.3 (text); top result: urban_air_quality_study.pdf p.1 (text, sim 0.8323)
- `t39` How many plots were used in the soil moisture trial? -> expected soil_moisture_trial.pdf p.1 (text); top result: borealis_annual_report.pdf p.4 (text, sim 0.811)
- `t42` Who funded the sleep and memory study? -> expected sleep_memory_study.pdf p.3 (text); top result: borealis_annual_report.pdf p.4 (text, sim 0.8226)
- `t43` How many days of paid annual leave do full-time employees receive? -> expected employee_handbook.pdf p.1 (text); top result: borealis_annual_report.pdf p.4 (text, sim 0.7895)
- `t45` What is the annual learning budget per employee? -> expected employee_handbook.pdf p.4 (text); top result: urban_air_quality_study.pdf p.1 (text, sim 0.8433)
- `t48` Within how many days are approved expense claims reimbursed? -> expected travel_expense_policy.pdf p.3 (text); top result: employee_handbook.pdf p.1 (text, sim 0.8754)
- `t50` How quickly must lost or stolen devices be reported? -> expected it_security_guidelines.pdf p.2 (text); top result: borealis_annual_report.pdf p.4 (text, sim 0.8455)
- `t51` Which encryption standard must company laptops use? -> expected it_security_guidelines.pdf p.2 (text); top result: urban_air_quality_study.pdf p.1 (text, sim 0.7808)

## Suggested thresholds

`TEXT_MIN_SIM=0.825` and `IMAGE_MIN_SIM=0.26` (on this question set: expected source kept 40%, easy off-topic blocked 100%; 5 threshold pairs scored equally well). Tuned on the same questions it is reported on, so treat it as a starting point.
