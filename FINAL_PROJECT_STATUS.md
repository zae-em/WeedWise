# Final project status

This report describes the original V1 training release. See the
[V2 completion report](FINAL_EXPANDED_PROJECT_COMPLETION.md) for the expanded model and its application verification.

Verified 2026-10-31T18:27:24.592544+00:00. Project: `D:\Haroon FYP\AI-Weed-Assistant`.

**All achievable requirements for the local 12-class research application are implemented and verified.**
Full GPU training and genuine held-out evaluation finished; the application uses the downloaded trained model.
This does not claim the proposal's larger species coverage, independent field validation or production hosting.

## Completed components

| Component | Actual result | Evidence |
| --- | --- | --- |
| Windows / VS Code | Python environment, pinned dependencies, local notebook kernel and official Colab extension | [Installation](docs/INSTALLATION.md), [environment checks](evaluation/setup_verification.json) |
| Dataset | CRC/SHA validation, preserved originals, 5,536 processed images, 12 classes, grouped 3883/827/826 split | [Dataset report](evaluation/dataset/DATASET_REPORT.md), [manifest](dataset/processed/manifest.json) |
| Full MobileNetV2 training | Tesla T4, 5 head + 20 fine-tuning epochs, full training split each epoch | [Actual provenance](evaluation/mobilenet_v2/run_provenance.json), [history](models/mobilenet_v2/history.json) |
| Local trained artifacts | Export, validation-selected best checkpoint, final recovery checkpoint, intermediate snapshot and original output archive | [Models](models/README.md) |
| Evaluation | Top-1/top-5, precision/recall/F1, per-class results, confusion matrix, curves, calibration, timing and actual Grad-CAM | [Model card](MODEL_CARD.md), [evaluation](evaluation/mobilenet_v2/EVALUATION_REPORT.md) |
| Local model verification | All 826 held-out images reproduce GPU accuracy on CPU; export matches selected best weights and final split | [Verification JSON](evaluation/local_model_verification.json) |
| Identification | Upload, camera widget, quality checks, genuine top-5/confidence/scientific names, crop/weed score, review gate and Grad-CAM | [Browser checks](evaluation/browser_e2e.json), [inference](weed_assistant/inference.py) |
| Knowledge / management | 12 species, Tier A/B, 31 references, claim-level citations, cultural/mechanical/integrated and verified group/resistance notes | [Knowledge](knowledge_base/README.md), [records](knowledge_base/species.json) |
| Chat assistant | Functional local extractive retrieval; species-scoped claims/references and unsupported/dose refusal | [Chatbot](weed_assistant/chatbot.py), [tests](evaluation/setup_tests.txt) |
| SQLite / review | Saved images, history, feedback, review decisions, CSV export, species records and separate sourced correction drafts | [Database](weed_assistant/database.py), [browser checks](evaluation/browser_e2e.json) |
| Streamlit interface | Seven pages, actual evaluation dashboard, desktop and mobile browser validation | [Application](app/main.py), [usage](docs/USAGE.md) |
| Documentation | Installation, usage, architecture, training/recovery, dataset/model/evaluation, deployment and limitations | [README](README.md), [architecture](docs/ARCHITECTURE.md), [deployment](docs/DEPLOYMENT.md) |

## Genuine results

- Test top-1: **93.10%**; top-5: **99.76%**, on **826** held-out images.
- Macro precision / recall / F1: 0.9298 / 0.9406 / 0.9319; weighted F1: 0.9332.
- Best selected epoch: **24 of 25** (one-based). CUDA job duration including evaluation: 15.75 minutes.
- Model-only median inference: T4 4.82 ms; local CPU 29.12 ms.
- Automated tests: **38 passed, zero failures, zero errors, zero skips**.
- Live Microsoft Edge workflows: **13 passed**, real model used, zero JavaScript errors.
- Full held-out CPU recheck: PASS; maximum probability difference from GPU 0.00001234.

See [model card](MODEL_CARD.md) for per-class performance and confidence interpretation. Test ECE increased
after validation-fitted scaling; no test-data recalibration was performed. Black-grass F1 is only 0.6727.

## Run and use

Open **http://127.0.0.1:8501** while the local server is running. To start it again:

```powershell
Set-Location 'D:\Haroon FYP\AI-Weed-Assistant'
.\.venv\Scripts\python.exe -m streamlit run app/main.py
```

Colab is no longer required for local inference. Models, reports, images and SQLite stay on the laptop;
no Google Drive or permanent cloud storage was used. Saved browser-test feedback is explicitly marked automated.
The historical setup reports are superseded by this status.

## Explicit limits and remaining external validation

- Available coverage is **nine weeds + three crops**, not 150-250 weed species + five crops.
- No independent field or labeled unknown-species dataset was available. Low confidence flags review;
  unsupported plants can receive high confidence. No unknown-detection performance is claimed.
- Exact/near duplicates and detected source-frame families do not cross splits. Missing longitudinal plant
  IDs prevent ruling out all related observations. Background/ruler shortcuts remain possible.
- Botanical sources are attributed and checked, but Pakistani suitability and local chemical approvals are
  not established. Expert-review needs remain in [BOTANICAL_REVIEW.json](knowledge_base/BOTANICAL_REVIEW.json).
- Camera UI renders, but physical capture/permission was not exercised in headless testing. No hardware-camera
  success is claimed. The application is verified as a local research prototype, not public production hosting.
- Optional EfficientNet-B0 comparison was not run. It is outside the required completed MobileNetV2 pipeline.

No essential local application functionality remains blocked. Expert field validation, new labeled species and
device-camera verification require evidence outside the completed software checks. See [limitations](docs/LIMITATIONS.md).

## Expansion recovery

See [RESUME_STATUS.md](RESUME_STATUS.md) for the current verified download, preprocessing, Colab and release checkpoints. [FINAL_EXPANDED_PROJECT_COMPLETION.md](FINAL_EXPANDED_PROJECT_COMPLETION.md) tracks release evidence.
