"""Write an evidence-based status report; unavailable v2 measurements remain pending."""
import json
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read(name,default=None):
    path=ROOT/name
    return json.loads(path.read_text()) if path.exists() else default


def main():
    source=read('dataset/cropandweed/source_manifest.json',{})
    raw=ROOT/'dataset/cropandweed/huggingface/hub/Voxel51/CropAndWeed'
    present=[]
    for entry in source.get('files',[]):
        path=raw/entry['path']
        if path.is_file() and path.stat().st_size==entry['bytes']:present.append(entry)
    downloaded=sum(r['bytes'] for r in present)
    state=read('dataset/cropandweed/download_status.json',{})
    original=read('evaluation/local_model_verification.json',{})
    old_kb=read('knowledge_base/species.json',{'species':[]})['species']
    taxonomy=read('dataset/cropandweed/taxonomy.json',{'labels':{}})['labels']
    all_candidates={r['class_label']:r for r in taxonomy.values() if r['included']}
    annotation=read('evaluation/cropandweed_annotation_audit.json',{})
    candidates={label:row for label,row in all_candidates.items() if not annotation or label in annotation['eligible_present_species']}
    old_classes={r['class_label'] for r in old_kb}
    overlaps=sorted(set(candidates)&old_classes)
    new={role:sorted(c for c,r in candidates.items() if c not in old_classes and r['role']==role) for role in ['weed','crop']}
    summary=read('dataset/combined_v2/summary.json',{})
    training=read('models/v2/training_status.json',{})
    metrics=read('evaluation/v2/test_metrics.json',{})
    local=read('evaluation/v2/local_verification.json',{})
    tests=(ROOT/'evaluation/expansion_tests.txt').read_text(errors='replace') if (ROOT/'evaluation/expansion_tests.txt').exists() else ''
    test_match=re.search(r'Ran (\d+) tests in ([\d.]+)s',tests)
    tests_status=(test_match.group(0)+'; '+next((line for line in tests.splitlines() if line.startswith('OK') or line.startswith('FAILED')),'result pending')) if test_match else 'Pending'
    browser=read('evaluation/ui/browser_e2e.json',{})
    fmt=lambda value: f'{value:.2%}' if value is not None else 'Pending: genuine trained v2 artifact unavailable'
    lines=['# Expanded Model Report','',f"Status updated: {datetime.now(timezone(timedelta(hours=5))).isoformat()} (Asia/Karachi).",'',
        '**Expansion is incomplete until the full data pipeline, GPU training, evaluation and local release checks finish.**',
        'The original working v1 model remains active. Prepared scripts, successful baseline tests or a GPU probe are not evidence of completed v2 training.','',
        '| Required deliverable | Verified status |','| --- | --- |',
        f"| 1. Dataset download | {state.get('state','not started')}; {len(present):,}/{source.get('file_count',0):,} files present at expected size. Full integrity audit runs after download. |",
        f'| 2. Downloaded dataset size | {downloaded:,} bytes present; expected complete repository {source.get("repository_bytes",0):,} bytes. Temporary/cache files are excluded. |',
        '| 3. Original species | 12 classes: 9 weeds and 3 crops. |',
        f"| 4. Newly added weeds | {len(summary['new_weeds']) if summary else 'Pending usable-instance and split audit'}; {len(new['weed'])} taxonomy candidates. |",
        f"| 5. Newly added crops | {len(summary['new_crops']) if summary else 'Pending usable-instance and split audit'}; {len(new['crop'])} taxonomy candidates. |",
        f"| 6. Final species | {summary.get('class_count','Pending; no fixed species total promised')} |",
        f"| 7. Training / validation / test images | {summary.get('counts','Pending full instance extraction and group-aware split')} |",
        f"| 8. Training epochs | {training.get('epochs',0)} verified v2 epochs |",
        f"| 9. Training duration | {training.get('elapsed_seconds','Not measured: v2 training not completed')} |",
        f"| 10. GPU | Tesla T4 verified earlier through VS Code with CUDA tensor result 3680.0; selected runtime must pass the current-session probe before training. |",
        f"| 11. Top-1 | {fmt(metrics.get('top1'))} |",
        f"| 12. Top-5 | {fmt(metrics.get('top5'))} |",
        f"| 13. Original-class performance | {metrics.get('groups',{}).get('original_heldout','Pending v2 evaluation; v1 baseline 93.10% / 99.76% top-1 / top-5 on 826 images')} |",
        f"| 14. New-class performance | {metrics.get('groups',{}).get('new_classes','Pending genuine v2 test predictions')} |",
        '| 15. Model weights | Original `models/mobilenet_v2/model.pt`; verified backup `models/v1/models/mobilenet_v2/model.pt`. V2 destination `models/v2/`; no trained v2 weights claimed until retrieved. |',
        '| 16. Application integration | Dynamic species counts, artifact crop/weed roles, versioned release selection, immutable model provenance, matching evaluation paths and cited knowledge-release builder implemented. V1 stays active; real v2 end-to-end checks remain pending. |',
        f'| 17. Automated tests | {tests_status}. Browser: {browser.get("status","Pending")}, {len(browser.get("steps",[]))} completed workflows; see evidence below. |',
        '| 18. Remaining limitations | Full download, final retained species/splits, genuine GPU training/evaluation, final local v2 release and expanded browser verification remain required. Austrian-to-Pakistani domain shift and unknown-species detection are unvalidated. |','',
        '## Source, licence and integrity','',
        '- Source: https://huggingface.co/datasets/Voxel51/CropAndWeed',
        f"- Pinned commit: `{source.get('revision','unresolved')}`. Metadata export version: 1.18.0.",
        '- Full export: 8,034 field images and corresponding semantic masks; `samples.json` contains 8,034 source records. No preview subset is substituted.',
        '- Custom non-commercial research licence; images and annotations must not be redistributed. See https://github.com/cropandweed/cropandweed-dataset/blob/main/LICENCE .',
        '- Official resumable Hub snapshot client plus FiftyOne `load_from_hub`/local export import. Exact file sizes, LFS SHA256 values and Git object IDs are saved in `dataset/cropandweed/source_manifest.json`.',
        '- FiftyOne 1.22.1 is installed in `.venv-data`; `pip check` and the actual import check passed. Initial network timeouts and one rejected cached-package hash mismatch were resolved by a clean-cache installation; hash verification was not disabled.',
        '- Full dataset integrity has not passed until `dataset/cropandweed/download_status.json` records the completed audit.','',
        '## Original species and preservation','',
        '| Dataset label | Scientific name | Role |','| --- | --- | --- |']
    lines += [f"| {r['class_label']} | {r['scientific_name']} | {r['role']} |" for r in old_kb]
    lines += ['',f"Original exported model SHA256: `{original.get('model_sha256','unavailable')}`.",
        'Original 25-epoch evidence: 93.09927360774818% top-1, 99.75786924939467% top-5 on 826 held-out images. The v1 backup manifest records hashes for model files and reports. Original datasets, manifest, class mapping and knowledge records remain intact.',
        '', '## Taxonomy candidates, not trained support','',
        f'Unique resolved taxonomy definitions: {len(all_candidates)}; eligible species actually present in annotations: {len(candidates)}. Growth stages are merged and generic/unresolved labels are excluded.',
        f"The complete annotation audit verified {annotation.get('source_images','pending')} images, {annotation.get('recording_sessions','pending')} recording sessions and {annotation.get('total_bounding_boxes','pending')} bounding boxes with {len(annotation.get('issues',[]))} label/session metadata issues. Of these, {annotation.get('empty_detection_images','pending')} images have no detections. SHA256 of samples.json matched the pinned Hub manifest.",
        'Scientific/common associations require saved exact GBIF species and vernacular evidence. Crop/weed roles follow the authors explicit simplified mappings; unresolved roles are excluded rather than guessed.',
        'Sugar-beet cultivated-variety and parent-species labels merge into the original Sugar beet output. Duplicate Hedge mustard IDs merge at species rank.',
        '', '**Overlapping candidate classes:** '+', '.join(overlaps)+'.',
        '', '**New weed candidates:** '+', '.join(new['weed'])+'.',
        '', '**New crop candidates:** '+', '.join(new['crop'])+'.',
        '', 'These lists can shrink after crop-quality and independent session/split checks. They do not establish supported classes or an accuracy claim.',
        '', '## Implemented pipeline and recovery','',
        '- Preserve original pixel hashes and every original split assignment. Crop field images using target bounding boxes and semantic masks; reject tiny, mixed-species and overlapping instances. Preserve source-image, recording-set, session, raw annotation and crop hashes.',
        '- Detect exact duplicates and near duplicates of both crops and full source images. Union recording sessions, source images and duplicate components before splitting; quarantine new components that collide with incompatible frozen original splits.',
        '- Preserve all original class outputs and backbone weights; initialize new outputs, replay underrepresented classes while covering every training image, and distil original predictions to reduce forgetting.',
        '- AdamW, augmentation, head/fine-tuning phases, validation scheduling/early stopping, optimizer/RNG/scaler resumption, best/final exports and immutable five-epoch recovery snapshots are implemented.',
        '- Calibration uses validation only. Final evaluation produces top-1/top-5, macro precision/recall/F1, weighted and per-class F1, confusion matrix, curves, ECE and latency, with original/new/combined groups and original-held-out regression.',
        '- No CPU fallback for expanded training. No GPU epochs or expanded metrics are claimed from unit-test fixtures.',
        '- Colab files are temporary. Recovery snapshots become safe from runtime deletion only after a verified local transfer. Browser downloads may need user action; no always-on checkpoint-delivery guarantee is made.',
        '', '## Colab execution status','',
        'The installed project controller uses supported VS Code notebook APIs and the official Colab filesystem provider. A genuine Tesla T4 calculation succeeded; subsequent filesystem transfers failed and the extension requested reauthentication. Notebook chunk transport is prepared, but its current-session execution must still be verified.',
        'The user opened a browser Colab notebook. A browser scratchpad at `https://colab.research.google.com/notebooks/empty.ipynb` provides temporary code without saving edits to Drive. This browser session genuinely printed CUDA: True and Tesla T4, and showed Cannot save changes. Datasets and model files are never mounted or uploaded to Drive by the worker.',
        'The browser upload widget remained disabled and the runtime repeatedly reconnected. Native Files-pane upload was attempted, but completion and archive integrity have not been established. A manual complete upload of training/expanded_colab_inputs.zip is currently requested. No expansion worker or training epoch has been verified started.',
        'Colab File > Upload notebook unexpectedly created a notebook shell in Drive. Only source code was in that shell; no images or weights were uploaded to Drive. This incidental shell requires cleanup or explicit user retention; it is not used as permanent project storage.',
        '', '## Application and knowledge limits','',
        'The existing app remains usable with v1. V2 activation requires genuine completed weights, full local test-inference comparison, matching mappings and a source-backed knowledge file. Prediction history and feedback are preserved; a new model-version table records immutable mappings by hash.',
        'New taxonomy records are generated only for retained trained classes. Species-specific management information that lacks verified evidence is explicitly unavailable; the chatbot must decline unsupported advice. Existing sourced cultural, mechanical, integrated and herbicide-group guidance is retained. No pesticide product, dose or mixture is invented.',
        '', '## Executed verification','',
        f'- Python: {tests_status}; `evaluation/expansion_tests.txt` contains full output. Three real-v2 checks are explicitly skipped while weights/data are absent; they are not counted as executed validation of an expanded model.',
        f'- Browser: {browser.get("status","Pending")}, {len(browser.get("steps",[]))} completed workflows in `evaluation/ui/browser_e2e.json` and `evaluation/expansion_browser_final.log`.',
        '- Two browser-selector/interaction issues were corrected: the normalization widget now has role switch, and deletion confirmation requires Enter to commit the input. Failed-run test records were removed only after verifying their automated-test marker; genuine history was retained.',
        '- Protected original model, evaluation artifacts, dataset manifest and knowledge hashes remained unchanged in the completed browser checks.',
        '- Compiled Python sources and verified actual FiftyOne import and dependency consistency.',
        '', '## Continue and deliver','',
        'See README.md for the local download supervisor, supported Colab launch/monitor workflow, browser scratchpad fallback, recovery and local release commands. Final v2 publication and updated measurements must wait for successful execution and retrieval, not merely notebook preparation.','']
    (ROOT/'EXPANDED_MODEL_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Updated EXPANDED_MODEL_REPORT.md:',len(present),'files,',downloaded,'bytes;',tests_status)


if __name__=='__main__':main()
