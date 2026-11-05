# Expanded Model Report — pre-training record

This report records the 3 November preparation milestone. See [the completed V2 release report](FINAL_EXPANDED_PROJECT_COMPLETION.md) for the final training, evaluation and application results.

Status updated: 2026-11-03T19:47:09.662213+05:00 (Asia/Karachi).

**Expansion is incomplete until the full data pipeline, GPU training, evaluation and local release checks finish.**
The original working v1 model remains active. Prepared scripts, successful baseline tests or a GPU probe are not evidence of completed v2 training.

| Required deliverable | Verified status |
| --- | --- |
| 1. Dataset download | complete; 16,075/16,075 files present at expected size. Full integrity audit runs after download. |
| 2. Downloaded dataset size | 11,646,405,955 bytes present; expected complete repository 11,646,405,955 bytes. Temporary/cache files are excluded. |
| 3. Original species | 12 classes: 9 weeds and 3 crops. |
| 4. Newly added weeds | 31; 37 taxonomy candidates. |
| 5. Newly added crops | 5; 6 taxonomy candidates. |
| 6. Final species | 48 |
| 7. Training / validation / test images | {'train': 21967, 'val': 5822, 'test': 5322} |
| 8. Training epochs | 0 verified v2 epochs |
| 9. Training duration | Not measured: v2 training not completed |
| 10. GPU | Tesla T4 verified earlier through VS Code with CUDA tensor result 3680.0; selected runtime must pass the current-session probe before training. |
| 11. Top-1 | Pending: genuine trained v2 artifact unavailable |
| 12. Top-5 | Pending: genuine trained v2 artifact unavailable |
| 13. Original-class performance | Pending v2 evaluation; v1 baseline 93.10% / 99.76% top-1 / top-5 on 826 images |
| 14. New-class performance | Pending genuine v2 test predictions |
| 15. Model weights | Original `models/mobilenet_v2/model.pt`; verified backup `models/v1/models/mobilenet_v2/model.pt`. V2 destination `models/v2/`; no trained v2 weights claimed until retrieved. |
| 16. Application integration | Dynamic species counts, artifact crop/weed roles, versioned release selection, immutable model provenance, matching evaluation paths and cited knowledge-release builder implemented. V1 stays active; real v2 end-to-end checks remain pending. |
| 17. Automated tests | Ran 62 tests in 15.918s; OK (skipped=3). Browser: PASS, 24 completed workflows; see evidence below. |
| 18. Remaining limitations | Full download, final retained species/splits, genuine GPU training/evaluation, final local v2 release and expanded browser verification remain required. Austrian-to-Pakistani domain shift and unknown-species detection are unvalidated. |

## Source, licence and integrity

- Source: https://huggingface.co/datasets/Voxel51/CropAndWeed
- Pinned commit: `d8e3000678d2307fff6834e110e9b2941848b2fd`. Metadata export version: 1.18.0.
- Full export: 8,034 field images and corresponding semantic masks; `samples.json` contains 8,034 source records. No preview subset is substituted.
- Custom non-commercial research licence; images and annotations must not be redistributed. See https://github.com/cropandweed/cropandweed-dataset/blob/main/LICENCE .
- Official resumable Hub snapshot client plus FiftyOne `load_from_hub`/local export import. Exact file sizes, LFS SHA256 values and Git object IDs are saved in `dataset/cropandweed/source_manifest.json`.
- FiftyOne 1.22.1 is installed in `.venv-data`; `pip check` and the actual import check passed. Initial network timeouts and one rejected cached-package hash mismatch were resolved by a clean-cache installation; hash verification was not disabled.
- Full dataset integrity has not passed until `dataset/cropandweed/download_status.json` records the completed audit.

## Original species and preservation

| Dataset label | Scientific name | Role |
| --- | --- | --- |
| Black-grass | Alopecurus myosuroides | weed |
| Charlock | Sinapis arvensis | weed |
| Cleavers | Galium aparine | weed |
| Common Chickweed | Stellaria media | weed |
| Common wheat | Triticum aestivum | crop |
| Fat Hen | Chenopodium album | weed |
| Loose Silky-bent | Apera spica-venti | weed |
| Maize | Zea mays | crop |
| Scentless Mayweed | Tripleurospermum inodorum | weed |
| Shepherds Purse | Capsella bursa-pastoris | weed |
| Small-flowered Cranesbill | Geranium pusillum | weed |
| Sugar beet | Beta vulgaris subsp. vulgaris var. altissima | crop |

Original exported model SHA256: `5e8f8943b7bb162c2b2a2196273da50bbb398b3a191bf5ee6d6b9a8a8d044f17`.
Original 25-epoch evidence: 93.09927360774818% top-1, 99.75786924939467% top-5 on 826 held-out images. The v1 backup manifest records hashes for model files and reports. Original datasets, manifest, class mapping and knowledge records remain intact.

## Taxonomy candidates, not trained support

Unique resolved taxonomy definitions: 53; eligible species actually present in annotations: 50. Growth stages are merged and generic/unresolved labels are excluded.
The complete annotation audit verified 8034 images, 929 recording sessions and 111953 bounding boxes with 0 label/session metadata issues. Of these, 234 images have no detections. SHA256 of samples.json matched the pinned Hub manifest.
Scientific/common associations require saved exact GBIF species and vernacular evidence. Crop/weed roles follow the authors explicit simplified mappings; unresolved roles are excluded rather than guessed.
Sugar-beet cultivated-variety and parent-species labels merge into the original Sugar beet output. Duplicate Hedge mustard IDs merge at species rank.

**Overlapping candidate classes:** Cleavers, Common Chickweed, Fat Hen, Maize, Shepherds Purse, Small-flowered Cranesbill, Sugar beet.

**New weed candidates:** Agrostemma githago, Amaranthus retroflexus, Ballota nigra, Barbarea vulgaris, Bromus hordeaceus, Bromus secalinus, Calepina irregularis, Centaurea cyanus, Cirsium arvense, Convolvulus arvensis, Datura stramonium, Digitaria sanguinalis, Echinochloa crus-galli, Fagopyrum esculentum, Fallopia convolvulus, Fallopia dumetorum, Fumaria officinalis, Galinsoga parviflora, Hordeum murinum, Lactuca serriola, Lamium purpureum, Matricaria chamomilla, Matricaria discoidea, Mentha piperita, Papaver rhoeas, Persicaria maculosa, Plantago lanceolata, Plantago major, Poa annua, Poa trivialis, Polygonum aviculare, Setaria viridis, Sisymbrium officinale, Solanum nigrum, Spergula arvensis, Thlaspi arvense, Veronica hederifolia.

**New crop candidates:** Glycine max, Helianthus annuus, Lathyrus oleraceus, Phaseolus vulgaris, Solanum tuberosum, Vicia faba.

These lists can shrink after crop-quality and independent session/split checks. They do not establish supported classes or an accuracy claim.

## Implemented pipeline and recovery

- Preserve original pixel hashes and every original split assignment. Crop field images using target bounding boxes and semantic masks; reject tiny, mixed-species and overlapping instances. Preserve source-image, recording-set, session, raw annotation and crop hashes.
- Detect exact duplicates and near duplicates of both crops and full source images. Union recording sessions, source images and duplicate components before splitting; quarantine new components that collide with incompatible frozen original splits.
- Preserve all original class outputs and backbone weights; initialize new outputs, replay underrepresented classes while covering every training image, and distil original predictions to reduce forgetting.
- AdamW, augmentation, head/fine-tuning phases, validation scheduling/early stopping, optimizer/RNG/scaler resumption, best/final exports and immutable five-epoch recovery snapshots are implemented.
- Calibration uses validation only. Final evaluation produces top-1/top-5, macro precision/recall/F1, weighted and per-class F1, confusion matrix, curves, ECE and latency, with original/new/combined groups and original-held-out regression.
- No CPU fallback for expanded training. No GPU epochs or expanded metrics are claimed from unit-test fixtures.
- Colab files are temporary. Recovery snapshots become safe from runtime deletion only after a verified local transfer. Browser downloads may need user action; no always-on checkpoint-delivery guarantee is made.

## Colab execution status

The installed project controller uses supported VS Code notebook APIs and the official Colab filesystem provider. A genuine Tesla T4 calculation succeeded; subsequent filesystem transfers failed and the extension requested reauthentication. Notebook chunk transport is prepared, but its current-session execution must still be verified.
The user opened a browser Colab notebook. A browser scratchpad at `https://colab.research.google.com/notebooks/empty.ipynb` provides temporary code without saving edits to Drive. This browser session genuinely printed CUDA: True and Tesla T4, and showed Cannot save changes. Datasets and model files are never mounted or uploaded to Drive by the worker.
The browser upload widget remained disabled and the runtime repeatedly reconnected. Native Files-pane upload was attempted, but completion and archive integrity have not been established. A manual complete upload of training/expanded_colab_inputs.zip is currently requested. No expansion worker or training epoch has been verified started.
Colab File > Upload notebook unexpectedly created a notebook shell in Drive. Only source code was in that shell; no images or weights were uploaded to Drive. This incidental shell requires cleanup or explicit user retention; it is not used as permanent project storage.

## Application and knowledge limits

The existing app remains usable with v1. V2 activation requires genuine completed weights, full local test-inference comparison, matching mappings and a source-backed knowledge file. Prediction history and feedback are preserved; a new model-version table records immutable mappings by hash.
New taxonomy records are generated only for retained trained classes. Species-specific management information that lacks verified evidence is explicitly unavailable; the chatbot must decline unsupported advice. Existing sourced cultural, mechanical, integrated and herbicide-group guidance is retained. No pesticide product, dose or mixture is invented.

## Executed verification

- Python: Ran 62 tests in 15.918s; OK (skipped=3); `evaluation/expansion_tests.txt` contains full output. Three real-v2 checks are explicitly skipped while weights/data are absent; they are not counted as executed validation of an expanded model.
- Browser: PASS, 24 completed workflows in `evaluation/ui/browser_e2e.json` and `evaluation/expansion_browser_final.log`.
- Two browser-selector/interaction issues were corrected: the normalization widget now has role switch, and deletion confirmation requires Enter to commit the input. Failed-run test records were removed only after verifying their automated-test marker; genuine history was retained.
- Protected original model, evaluation artifacts, dataset manifest and knowledge hashes remained unchanged in the completed browser checks.
- Compiled Python sources and verified actual FiftyOne import and dependency consistency.

## Continue and deliver

See README.md for the local download supervisor, supported Colab launch/monitor workflow, browser scratchpad fallback, recovery and local release commands. Final v2 publication and updated measurements must wait for successful execution and retrieval, not merely notebook preparation.
