"""Write completion documentation only after genuine local verification and application checks pass."""
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read(name):return json.loads((ROOT/name).read_text(encoding="utf-8"))


def main():
    metrics=read("evaluation/mobilenet_v2/test_metrics.json")
    local=read("evaluation/local_model_verification.json")
    tests=read("evaluation/setup_tests.json")
    browser=read("evaluation/browser_e2e.json")
    job=read("evaluation/mobilenet_v2/job_status.json")
    provenance=read("evaluation/mobilenet_v2/run_provenance.json")
    calibration=read("evaluation/mobilenet_v2/calibration.json")
    history=read("models/mobilenet_v2/history.json")
    classes=read("models/mobilenet_v2/classes.json")
    class_report=read("evaluation/mobilenet_v2/classification_report.json")
    kb=read("knowledge_base/species.json")
    assert job["state"]=="complete" and job["gpu"]=="Tesla T4"
    assert local["status"]=="PASS" and local["real_trained_model"] and local["test_samples"]==826
    assert tests["passed"] and not tests["skipped"] and tests["real_model_available"]
    assert browser["status"]=="PASS" and browser["realModel"] and not browser["errors"]
    with (ROOT/"models/mobilenet_v2/model.pt").open("rb") as stream:digest=hashlib.file_digest(stream,"sha256").hexdigest()
    assert digest==local["model_sha256"]
    best=history[local["best_epoch"]-1]
    generated=datetime.now(timezone.utc).isoformat()
    table="\n".join(f"| {label} | {class_report[label]['precision']:.4f} | {class_report[label]['recall']:.4f} | {class_report[label]['f1-score']:.4f} | {int(class_report[label]['support'])} |" for label in classes)
    card=f"""# MobileNetV2 model card

Generated from actual local artifacts: {generated}.

## Purpose and scope

Research identification support for 12 seedling dataset classes, comprising nine weed and three crop labels.
The Streamlit app runs the genuine export locally on Windows CPU. Adult plants, Pakistani fields, mixed-species
images and unsupported species are not independently validated. The model does not authorize field treatment.
The proposal target of 150-250 weed species plus five crops is not met by the provided dataset.

## Data and training

The supplied archive contained the same 5,539-image collection twice. Canonical originals were preserved;
three exact duplicate images were excluded, leaving 5,536 processed images at 224 x 224 RGB. Final grouped
splits are 3,883 training / 827 validation / 826 test. Exact/near duplicates and shared filename source-frame
families stay in one split. Longitudinal plant identities are unavailable, so related growth-stage observations
may remain across splits. See [dataset report](evaluation/dataset/DATASET_REPORT.md).

ImageNet-initialized MobileNetV2 trained for {len(history)} complete epochs on **{job['gpu']}**: five frozen-backbone
head epochs and twenty upper-backbone fine-tuning epochs (feature blocks 14 onward). Class-weighted cross
entropy, AdamW, augmentation, mixed precision, learning-rate scheduling and validation-loss early stopping were
enabled. Each epoch used all 3,883 training examples. Best validation loss selected **epoch {local['best_epoch']}
(one-based; stored checkpoint index {local['best_epoch']-1})**, with validation top-1 {best['val']['top1']:.4%}.
Last-epoch training top-1 was {metrics['training_accuracy_last_epoch']:.4%}; last validation top-1 was
{metrics['validation_accuracy_last_epoch']:.4%}. The job including evaluation took {job['duration_seconds']:.1f} seconds.
Test examples did not control training, model selection, calibration or review-threshold fitting.

## Measured held-out performance

| Measure | Result |
| --- | ---: |
| Test images | {metrics['test_samples']} |
| Top-1 accuracy | {metrics['top1']:.4%} |
| Top-5 accuracy | {metrics['top5']:.4%} |
| Macro precision | {metrics['precision_macro']:.6f} |
| Macro recall | {metrics['recall_macro']:.6f} |
| Macro F1 | {metrics['macro_f1']:.6f} |
| Weighted F1 | {metrics['weighted_f1']:.6f} |
| Crop/weed accuracy over supported labels | {metrics['crop_weed_accuracy_supported_classes']:.4%} |
| ECE before temperature scaling | {metrics['ece_before']:.6f} |
| ECE after temperature scaling | {metrics['ece_after']:.6f} |
| T4 median single-image model inference | {metrics['model_inference_median_ms']:.2f} ms |
| Windows CPU median single-image model inference | {local['median_cpu_model_inference_ms']:.2f} ms |

CPU timings are a local warm-model sample, not an end-to-end upload benchmark. The full local 826-image
verification reproduced top-1/top-5 results, with maximum probability difference from GPU
{local['max_probability_difference_from_gpu']:.8f}. Grad-CAM examples include real mistakes rather than only successes.

### Per-class held-out results

| Dataset class | Precision | Recall | F1 | Test images |
| --- | ---: | ---: | ---: | ---: |
{table}

Black-grass is the weakest class (F1 {class_report['Black-grass']['f1-score']:.4f}). In the confusion matrix,
28 Loose Silky-bent images were predicted as Black-grass and seven Black-grass images as Loose Silky-bent.
These grasses need particular caution. No model selection or further tuning used those test mistakes.

## Confidence and unknown plants

Temperature {calibration['temperature']:.6f} was fitted on 827 validation examples, improving validation NLL from
{calibration['validation_nll_before']:.6f} to {calibration['validation_nll_after']:.6f}. **Test ECE increased**
from {metrics['ece_before']:.6f} to {metrics['ece_after']:.6f}; calibration did not improve every test measure.
The temperature was not refitted using test data.

The review threshold {calibration['unknown_threshold']:.6f} is the fifth percentile of correctly classified
validation confidence. It flags {local['known_test_images_flagged_by_confidence']} known-class test images;
acceptance is {metrics['known_class_acceptance_rate']:.2%}, with {metrics['accuracy_on_accepted']:.2%} accuracy among
accepted known-class images. This is **not unknown-species detection accuracy**. No labeled unknown-species
or independent external field test set was available. Quality flags can additionally request review.

## Artifacts and reproducibility

- Export: `models/mobilenet_v2/model.pt`; SHA256 `{digest}`.
- Selected/final recovery checkpoints: `best.pt` / `last.pt`; intermediate snapshot is preserved separately.
- Exact executed code bundle: `training/trained_run_source.zip`.
- Actual configuration, split, history and CUDA package freeze accompany the model.
- Runtime: Python {provenance['python'].split()[0]}, torch {provenance['versions']['torch']},
  torchvision {provenance['versions']['torchvision']}; inference verified under the pinned Windows environment.
- [Measured evaluation](evaluation/mobilenet_v2/EVALUATION_REPORT.md),
  [local verification](evaluation/local_model_verification.json), [training guide](docs/TRAINING.md).

The dataset publisher is [Aarhus Plant Seedlings](https://vision.eng.au.dk/plant-seedlings-dataset/).
Redistribution rights must follow the original dataset/weight terms; this project does not claim a new license
for third-party data or weights. See [limitations](docs/LIMITATIONS.md) for acquisition, source and field-use limits.
"""
    (ROOT/"MODEL_CARD.md").write_text(card,encoding="utf-8")
    status=f"""# Final project status

Verified {generated}. Project: `D:\\Haroon FYP\\AI-Weed-Assistant`.

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
| Knowledge / management | 12 species, Tier A/B, {len(kb['sources'])} references, claim-level citations, cultural/mechanical/integrated and verified group/resistance notes | [Knowledge](knowledge_base/README.md), [records](knowledge_base/species.json) |
| Chat assistant | Functional local extractive retrieval; species-scoped claims/references and unsupported/dose refusal | [Chatbot](weed_assistant/chatbot.py), [tests](evaluation/setup_tests.txt) |
| SQLite / review | Saved images, history, feedback, review decisions, CSV export, species records and separate sourced correction drafts | [Database](weed_assistant/database.py), [browser checks](evaluation/browser_e2e.json) |
| Streamlit interface | Seven pages, actual evaluation dashboard, desktop and mobile browser validation | [Application](app/main.py), [usage](docs/USAGE.md) |
| Documentation | Installation, usage, architecture, training/recovery, dataset/model/evaluation, deployment and limitations | [README](README.md), [architecture](docs/ARCHITECTURE.md), [deployment](docs/DEPLOYMENT.md) |

## Genuine results

- Test top-1: **{metrics['top1']:.2%}**; top-5: **{metrics['top5']:.2%}**, on **826** held-out images.
- Macro precision / recall / F1: {metrics['precision_macro']:.4f} / {metrics['recall_macro']:.4f} / {metrics['macro_f1']:.4f}; weighted F1: {metrics['weighted_f1']:.4f}.
- Best selected epoch: **{local['best_epoch']} of {len(history)}** (one-based). CUDA job duration including evaluation: {job['duration_seconds']/60:.2f} minutes.
- Model-only median inference: T4 {metrics['model_inference_median_ms']:.2f} ms; local CPU {local['median_cpu_model_inference_ms']:.2f} ms.
- Automated tests: **{tests['tests_run']} passed, zero failures, zero errors, zero skips**.
- Live Microsoft Edge workflows: **{len(browser['steps'])} passed**, real model used, zero JavaScript errors.
- Full held-out CPU recheck: PASS; maximum probability difference from GPU {local['max_probability_difference_from_gpu']:.8f}.

See [model card](MODEL_CARD.md) for per-class performance and confidence interpretation. Test ECE increased
after validation-fitted scaling; no test-data recalibration was performed. Black-grass F1 is only 0.6727.

## Run and use

Open **http://127.0.0.1:8501** while the local server is running. To start it again:

```powershell
Set-Location 'D:\\Haroon FYP\\AI-Weed-Assistant'
.\\.venv\\Scripts\\python.exe -m streamlit run app/main.py
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
"""
    (ROOT/"FINAL_PROJECT_STATUS.md").write_text(status,encoding="utf-8")
    print("Wrote MODEL_CARD.md and FINAL_PROJECT_STATUS.md from passing genuine-model evidence.")


if __name__=="__main__":main()
