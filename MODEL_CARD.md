# MobileNetV2 model card

Generated from actual local artifacts: 2026-10-31T18:27:24.592544+00:00.

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

ImageNet-initialized MobileNetV2 trained for 25 complete epochs on **Tesla T4**: five frozen-backbone
head epochs and twenty upper-backbone fine-tuning epochs (feature blocks 14 onward). Class-weighted cross
entropy, AdamW, augmentation, mixed precision, learning-rate scheduling and validation-loss early stopping were
enabled. Each epoch used all 3,883 training examples. Best validation loss selected **epoch 24
(one-based; stored checkpoint index 23)**, with validation top-1 96.2515%.
Last-epoch training top-1 was 97.4504%; last validation top-1 was
95.4051%. The job including evaluation took 945.0 seconds.
Test examples did not control training, model selection, calibration or review-threshold fitting.

## Measured held-out performance

| Measure | Result |
| --- | ---: |
| Test images | 826 |
| Top-1 accuracy | 93.0993% |
| Top-5 accuracy | 99.7579% |
| Macro precision | 0.929793 |
| Macro recall | 0.940623 |
| Macro F1 | 0.931917 |
| Weighted F1 | 0.933188 |
| Crop/weed accuracy over supported labels | 99.5157% |
| ECE before temperature scaling | 0.022396 |
| ECE after temperature scaling | 0.026539 |
| T4 median single-image model inference | 4.82 ms |
| Windows CPU median single-image model inference | 29.12 ms |

CPU timings are a local warm-model sample, not an end-to-end upload benchmark. The full local 826-image
verification reproduced top-1/top-5 results, with maximum probability difference from GPU
0.00001234. Grad-CAM examples include real mistakes rather than only successes.

### Per-class held-out results

| Dataset class | Precision | Recall | F1 | Test images |
| --- | ---: | ---: | ---: | ---: |
| Black-grass | 0.5606 | 0.8409 | 0.6727 | 44 |
| Charlock | 0.9855 | 1.0000 | 0.9927 | 68 |
| Cleavers | 1.0000 | 0.9375 | 0.9677 | 48 |
| Common Chickweed | 0.9412 | 1.0000 | 0.9697 | 96 |
| Common wheat | 0.9444 | 0.9714 | 0.9577 | 35 |
| Fat Hen | 0.9750 | 0.9873 | 0.9811 | 79 |
| Loose Silky-bent | 0.9358 | 0.7727 | 0.8465 | 132 |
| Maize | 0.9730 | 0.9730 | 0.9730 | 37 |
| Scentless Mayweed | 0.9512 | 0.9070 | 0.9286 | 86 |
| Shepherds Purse | 0.9024 | 0.9487 | 0.9250 | 39 |
| Small-flowered Cranesbill | 0.9884 | 0.9884 | 0.9884 | 86 |
| Sugar beet | 1.0000 | 0.9605 | 0.9799 | 76 |

Black-grass is the weakest class (F1 0.6727). In the confusion matrix,
28 Loose Silky-bent images were predicted as Black-grass and seven Black-grass images as Loose Silky-bent.
These grasses need particular caution. No model selection or further tuning used those test mistakes.

## Confidence and unknown plants

Temperature 0.879227 was fitted on 827 validation examples, improving validation NLL from
0.123151 to 0.121496. **Test ECE increased**
from 0.022396 to 0.026539; calibration did not improve every test measure.
The temperature was not refitted using test data.

The review threshold 0.775177 is the fifth percentile of correctly classified
validation confidence. It flags 78 known-class test images;
acceptance is 90.56%, with 97.19% accuracy among
accepted known-class images. This is **not unknown-species detection accuracy**. No labeled unknown-species
or independent external field test set was available. Quality flags can additionally request review.

## Artifacts and reproducibility

- Export: `models/mobilenet_v2/model.pt`; SHA256 `5e8f8943b7bb162c2b2a2196273da50bbb398b3a191bf5ee6d6b9a8a8d044f17`.
- Selected/final recovery checkpoints: `best.pt` / `last.pt`; intermediate snapshot is preserved separately.
- Exact executed code bundle: `training/trained_run_source.zip`.
- Actual configuration, split, history and CUDA package freeze accompany the model.
- Runtime: Python 3.13.16, torch 2.11.0+cu130,
  torchvision 0.26.0+cu130; inference verified under the pinned Windows environment.
- [Measured evaluation](evaluation/mobilenet_v2/EVALUATION_REPORT.md),
  [local verification](evaluation/local_model_verification.json), [training guide](docs/TRAINING.md).

The dataset publisher is [Aarhus Plant Seedlings](https://vision.eng.au.dk/plant-seedlings-dataset/).
Redistribution rights must follow the original dataset/weight terms; this project does not claim a new license
for third-party data or weights. See [limitations](docs/LIMITATIONS.md) for acquisition, source and field-use limits.
