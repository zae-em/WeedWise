# AI Weed Identification & Management Assistant - Fieldwise

A local Streamlit application for **48 species (40 weeds and 8 crops)**, with calibrated top-5 suggestions,
source-referenced species records, a grounded local chatbot, image review and SQLite feedback.
MobileNetV2 was fine-tuned on a Colab Tesla T4. The application runs inference and stores observations locally.

**Verified V2 run:** 41 epochs with **69.67% top-1** and **93.91% top-5** accuracy on 5,322 held-out images.
See the [V2 completion report](FINAL_EXPANDED_PROJECT_COMPLETION.md) for results and validation limits.
The [original model card](MODEL_CARD.md) records the 12-class V1 baseline.

## Run locally

Install the [Windows Python environment](docs/INSTALLATION.md) from the repository root. A fresh clone contains
the source, notebooks and knowledge base. Trained weights, datasets and local records are separate artifacts.
Follow the [V2 workflow](#cropandweed-expansion-v2) to prepare the data, train or restore your model export,
and activate V2. The Examples tab uses local images from the prepared held-out split.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/main.py
```

Open **http://127.0.0.1:8501**.

## Application

- **Dashboard:** actual class/model/database statistics, recent activity, system status and working quick actions.
- **Identify plant:** upload, camera where available, or a genuine held-out example; quality checks, real top-5
  predictions, calibrated confidence, crop/weed aggregation, Grad-CAM and feedback. Save explicitly; retaining
  the photo is optional. Field crop selection records context without changing the model ranking.
- **Species explorer:** scientific/common search, role/family/type filters, class-linked profiles,
  sourced characteristics, references, Tier A detailed records and Tier B essential records.
- **Examples:** choose a species and test photo, run the active model locally, and compare its five ranked
  predictions with the dataset label. Open it from the dashboard or the Species Explorer tabs.
- **Weed management:** cultural, mechanical and integrated methods with regional sources. Herbicide information
  stays at verified group/resistance level, without brands, doses or mixing instructions.
- **AI chat assistant:** contextual species questions, cited local TF-IDF retrieval, suggested questions,
  comparisons, session conversation history and Clear chat. Unsupported questions are declined.
- **Prediction history:** search, species/date filters, retained photographs, details, CSV export and confirmed deletion.
- **Feedback & review:** review queues, corrected labels, reliable feedback, decisions and separate sourced drafts.
- **Model performance:** genuine metrics with interactive Plotly class results, confusion matrix, learning curves,
  confidence and latency. Original evaluation artifacts remain unchanged.
- **About:** implemented scope, data, architecture, methodology, technologies and field-use limitations.

Missing trained artifacts are reported explicitly; predictions are never replaced by random or mocked results.

## CropAndWeed expansion (v2)

The locally verified 48-class v2 model is active. The original model and reports are backed up under
`models/v1/`. Training, evaluation and release results are recorded in
[FINAL_EXPANDED_PROJECT_COMPLETION.md](FINAL_EXPANDED_PROJECT_COMPLETION.md).

The complete source is `Voxel51/CropAndWeed`, pinned to commit
`d8e3000678d2307fff6834e110e9b2941848b2fd`: **11,646,405,955 bytes**, including 8,034 photographs,
8,034 semantic masks, annotations and repository assets. Its custom non-commercial research licence
prohibits redistribution of images/annotations. Austrian photographs do not establish Pakistani field accuracy.

Data tools use a separate `.venv-data` environment; the working application environment is preserved.
Exact installed data dependencies are in `requirements-data.lock.txt`.

```powershell
.\.venv-data\Scripts\python.exe -u scripts/download_cropandweed.py
.\.venv\Scripts\python.exe scripts/resolve_cropandweed_taxonomy.py
.\.venv\Scripts\python.exe -m training.expansion_data
```

The official Hugging Face client resumes incomplete downloads under `dataset/cropandweed/`. Every repository
file is size/hash checked, images/masks are verified, and the complete export is loaded with FiftyOne.
Do not launch a second downloader while `evaluation/cropandweed_download.log` is advancing. The local
supervisor `scripts/local_expansion_pipeline.py` can finish preparation after the active download completes.
It never starts CPU training. Original pixels, labels, acquisition groups and split assignments are preserved.

Species resolution saves GBIF scientific/vernacular evidence, merges growth stages and original species overlaps
at species rank, and excludes generic or unresolved labels. The conversion isolates bounding boxes with their
semantic masks; rejects tiny, overlapping and mixed-species crops; detects exact/near duplicates of crops and
source images; and groups recording sessions before splitting. New species without independent train/validation/
test coverage are excluded and reported. All original 826 test images remain held out.

For the existing VS Code controller and official Colab extension:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_expansion_colab.py
.\.venv\Scripts\python.exe scripts/launch_expansion.py
.\.venv\Scripts\python.exe scripts/monitor_expansion.py
```

Select the already assigned T4 kernel and complete Google authentication yourself when required. A harmless
CUDA calculation is checked before use. The controller uses public VS Code notebook and filesystem APIs;
its notebook chunk transport is a fallback when the official filesystem provider fails. Neither transport
extracts credentials or uses private Colab APIs. The fallback still requires successful authentication and a
live selected kernel; preparation alone does not prove that it works on a disconnected session.

Browser fallback: `scripts/make_browser_expansion.py` prepares one input ZIP and the browser notebook.
**Colab's File > Upload notebook may automatically create a notebook shell in Drive.** For a workflow without
Drive notebook storage, use the built-in `https://colab.research.google.com/notebooks/empty.ipynb` scratchpad,
run the local `training/browser_entry.py` contents there, and choose `training/expanded_colab_inputs.zip`
when prompted. Do not choose Copy to Drive. Images and checkpoints are written only to temporary `/content`.
Keep the browser open for monitoring and download prompts. GPU selection is verified in code.

The v2 trainer loads the genuine v1 backbone and compatible head rows, adds new outputs, and uses full-coverage
balanced replay, augmentation, teacher distillation on original examples, AdamW, learning-rate scheduling,
validation early stopping, and resumable optimizer/RNG/scaler checkpoints. Validation reports original and new
classes separately. Test data is used only after model selection and validation-only calibration.

Immutable recovery snapshots are available every five epochs. They are protected against runtime deletion
**only after transfer to the laptop completes**. Browser fallback offers those files for download; VS Code
monitoring retrieves them with hashes. Reconnecting an expired runtime cannot recover files never downloaded.
Resume by placing a downloaded complete checkpoint in the runtime and setting `ExpansionConfig.resume` to it;
the saved dataset fingerprint and training configuration must match.

After the complete hashed artifact export has been retrieved:

```powershell
.\.venv\Scripts\python.exe scripts/import_expanded_artifacts.py 'path/to/expanded_artifacts.zip' --receipt 'path/to/expansion_transfer.json'
.\.venv\Scripts\python.exe scripts/release_expanded_model.py
.\.venv\Scripts\python.exe scripts/release_expanded_model.py --activate
```

The release step checks the full held-out CPU inference against GPU predictions, validates original split
preservation and class mappings, and builds matching cited species records. Activation selects v2 through
`models/active.json`; it never overwrites v1. Restart Streamlit after changing versions. The database records
immutable model hashes, class maps and versions while retaining old predictions and feedback. New species
with no verified management evidence explicitly show that gap; the chatbot declines unsupported guidance.

Run `.\.venv\Scripts\python.exe -m unittest discover -s tests -v` and, with Streamlit running,
`node tools/browser-test/ui_e2e.cjs`.

## Actual dataset

The supplied ZIP stored the same 5,539-image collection twice. All archive entries passed CRC checks and the
copies matched by SHA256. All canonical images decoded; three exact duplicates were excluded, leaving
**5,536 processed images**. Near-duplicates and shared filename source-image families stay in a shared split.
Original ZIPs and raw images are preserved.

| Split | Images |
| --- | ---: |
| Training | 3,883 |
| Validation | 827 |
| Test | 826 |

Coverage is **nine weed labels and three crop labels** (Common wheat, Maize, Sugar beet), rather than the
proposal's intended 150-250 weed species plus five crops. See the
[dataset report](evaluation/dataset/DATASET_REPORT.md) and [limitations](docs/LIMITATIONS.md).

## Training

Use [complete_training.ipynb](training/complete_training.ipynb) and the [training guide](docs/TRAINING.md).
Every optimization epoch uses the complete training split. The pipeline trains the frozen-backbone head,
then fine-tunes upper layers with augmentation, class weighting, AdamW, scheduling, early stopping and checkpoints.
Validation alone controls selection/calibration; test images remain held out until final evaluation.

The project VS Code controller uses supported notebook commands and the official Colab filesystem provider.
It does not extract authentication credentials. Authentication/runtime selection may still need your own action
following expiry. The full GPU job requires CUDA and has **no long CPU-training fallback**.

Runtime-only files are temporary. Only checkpoints actually downloaded to the laptop are protected from runtime
loss. `last.pt` embeds best weights and complete recovery state. The exact executed source bundle is preserved
as `training/trained_run_source.zip`.

## Key artifacts

| Path | Purpose |
| --- | --- |
| `models/mobilenet_v2/model.pt` | Genuine exported classifier and calibration |
| `models/mobilenet_v2/best.pt`, `last.pt` | Best weights and final recovery state |
| `models/checkpoints/intermediate.pt` | Preserved intermediate checkpoint |
| `evaluation/mobilenet_v2/` | Genuine GPU results, reports, plots and prediction arrays |
| `evaluation/local_model_verification.json` | Complete held-out CPU verification |
| `evaluation/setup_tests.json` | Automated tests, including any explicit skips |
| `evaluation/browser_e2e.json` | Live Edge browser workflow verification |
| `dataset/processed/manifest.json` | Authoritative final grouped splits and hashes |
| `knowledge_base/species.json` | Curated, source-attributed records |
| `app/data/weed_assistant.sqlite3` | Local history, feedback, reviews and species records |
| `app/uploads/` | Local copies of submitted photographs |
| `FINAL_PROJECT_STATUS.md` | Completion and verification evidence |
| `FINAL_EXPANDED_PROJECT_COMPLETION.md` | V2 training, evaluation and integration verification |
| `evaluation/ui/` | UI test results, browser captures and protected artifact hashes |

## Verify

```powershell
.\.venv\Scripts\python.exe scripts/check_ui.py
.\.venv\Scripts\python.exe scripts/verify_trained_model.py
node tools/browser-test/ui_e2e.cjs
```

The browser test requires the local server and uses existing Microsoft Edge in an isolated headless session.
Python tests explicitly skip genuine-model checks when no model exists; skips do not establish model validation.

See [usage](docs/USAGE.md), [model card](MODEL_CARD.md), [completion report](FINAL_PROJECT_STATUS.md),
[installation](docs/INSTALLATION.md), [architecture](docs/ARCHITECTURE.md), [training](docs/TRAINING.md),
[deployment scope](docs/DEPLOYMENT.md) and [limitations](docs/LIMITATIONS.md).

See [RESUME_STATUS.md](RESUME_STATUS.md) for the latest recovery checkpoint and [FINAL_EXPANDED_PROJECT_COMPLETION.md](FINAL_EXPANDED_PROJECT_COMPLETION.md) for verified release evidence.
