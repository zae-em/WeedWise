"""Create the final UI delivery report from completed test and artifact-preservation evidence."""
import json
from datetime import datetime,timezone
from pathlib import Path
from prepare_ui_upgrade import protected_hashes

ROOT=Path(__file__).resolve().parents[1]


def main():
    directory=ROOT/"evaluation/ui"
    tests=json.loads((directory/"tests.json").read_text())
    browser=json.loads((directory/"browser_e2e.json").read_text())
    before=json.loads((directory/"protected_artifacts_before.json").read_text())
    after=protected_hashes()
    assert before==after,"Protected ML/data/knowledge artifacts changed"
    assert tests["passed"] and not tests["skipped"] and tests["errors"]==tests["failures"]==0
    assert browser["status"]=="PASS" and not browser["errors"] and browser["testRecordDeleted"]
    (directory/"protected_artifacts_after.json").write_text(json.dumps(after,indent=2))
    preservation={"status":"PASS","protected_files":len(before),"weights_modified":False,"retrained":False,
                  "model_sha256":after["models/mobilenet_v2/model.pt"],"original_evaluation_unchanged":True,
                  "dataset_manifest_unchanged":True,"knowledge_records_unchanged":True}
    (directory/"preservation.json").write_text(json.dumps(preservation,indent=2))
    report=f"""# UI completion report

Verified {datetime.now(timezone.utc).isoformat()}. Application: **Fieldwise - AI Weed Identification & Management Assistant**.

The redesigned nine-page local application is implemented, integrated with the existing trained model and
verified through actual browser interactions. It runs at **http://127.0.0.1:8501**. The model was not retrained.

## Implemented pages and interactions

| Page | Working features |
| --- | --- |
| Dashboard | Welcome/hero actions, six shortcuts, actual class/role/accuracy/save/feedback statistics, recent SQLite activity and model/knowledge/chat/database status |
| Identify plant | File upload, camera widget, genuine held-out example, preview/dimensions, quality/error recovery, remove/replace, optional field crop context, real inference, calibrated five-candidate chart/bars/profile controls, uncertainty warnings, explicit saving, optional photo retention, Grad-CAM, feedback, species/management/chat links |
| Species explorer | All 12 model-linked records, scientific/common/alias search, weed/crop/family/type filters, profile cards, details, life cycle, characteristics, citations and context-linked actions |
| Weed management | Supported-weed selection, botanical context, sourced cultural/mechanical/integrated panels, verified chemical-group/resistance information, explicit unavailable sections, safety/source panels and contextual chat |
| AI chat assistant | Message bubbles, session history, Clear chat, suggested questions, actual identified-species context, cited local retrieval, supported-species comparison and unsupported/dose refusal |
| Prediction history | SQLite records, text/ID search, species/UTC date/flag filters, retained photos where available, details, latest feedback/corrected label, filtered CSV, confirmed individual deletion and clear-all confirmation |
| Feedback & review | Pending/reviewed/low-confidence/flag queues, saved review decisions, correct/incorrect/unsure feedback, supported corrections/comments, feedback counts and separate source-backed species draft records |
| Model performance | Actual architecture/epochs/classes/test counts, top-1/top-5/precision/recall/F1, interactive class metrics and count/normalized confusion matrix, learning curves, confidence histogram, calibration, GPU/CPU timing and original Grad-CAM |
| About | Implemented objectives/users, model/dataset/methodology/technologies, supported coverage, storage/privacy and field-use limitations |

## UI architecture and design

`app/main.py` owns nine-page routing, page URL state, sidebar and the error boundary. `app/ui.py` provides
shared views and cached services. `app/views/` separates page implementations. `app/style.css` supplies the
forest-green, sage and warm-neutral system, consistent cards, buttons, spacing, accessible native controls,
material icons and responsive layouts. Fonts are local system fonts; no new frontend framework is required.
The decorative dashboard SVG is an abstract illustration, not a claimed species photograph. Library cards
explicitly avoid plant photos without established redistribution rights.

Model and knowledge loading are cached. Widget changes do not reconstruct the neural network. JSON/NPZ
evaluation reads are cached by file version. SQLite uses indexed/parameterized queries and SQL aggregates.
Saved observations persist in SQLite; navigation and prediction/chat context use session state. The selected
page is encoded in the URL so reloading can restore that page. Reloading does not promise browser-session
prediction/chat persistence. Views show up to the latest 2,000 matching history/review records.

## Integration and preservation

- The unchanged `PlantClassifier` uses the original 224px preprocessing, class mapping, temperature and review
  threshold. Results and confidence are real inference outputs, not demonstration predictions.
- Field crop selection records context only. Rice, cotton and sugarcane are not added model classes, and no
  unvalidated crop-aware reranking is applied.
- `IdentificationService` supports session-only inference, followed by explicit idempotent saving. Retaining
  the photograph is optional and off by default. Feedback saves its linked result if necessary.
- SQLite schema and existing records are preserved. New indexes support filtering; no destructive schema
  migration was needed. Deletion removes explicit record IDs and dependent feedback/reviews transactionally.
  Photo cleanup is restricted to unreferenced files in the app uploads folder. Source dataset files are untouched.
- The original app and a consistent SQLite snapshot were backed up before changes; see [backup evidence](evaluation/ui/backup.json).
- **{len(before)} protected files are byte-identical**: all saved model/checkpoint weights, original model evaluation,
  dataset split manifest and curated knowledge. Export SHA256: `{preservation['model_sha256']}`.
- Original measured results remain **93.10% top-1 / 99.76% top-5 on 826 held-out images**. No optimization,
  retraining, new ML evaluation or weight modification was performed for this UI upgrade.

## Executed verification

- **{tests['tests_run']} automated tests passed**, zero failures, errors or skips. Includes the existing suite,
  all nine pages, real-model inference, SQLite filtering/latest feedback, photo-safe deletion, explicit saving,
  unchanged crop rankings, context/comparison grounding, URL actions and protected-artifact hashes.
- **{len(browser['steps'])} Microsoft Edge browser workflows passed**, zero JavaScript errors. A real held-out
  Fat Hen image was uploaded and processed. The reported suggestion was **{browser['predictedSpecies']}**;
  no expected confidence was hardcoded. Desktop, 820px tablet and 390px mobile interactions actually ran.
- Browser checks cover upload rejection/recovery, top five, explicit saving, feedback, Grad-CAM, contextual
  chat/refusal/clear/suggestions, library search/filter/details, management sources, history/export/confirmation,
  review saving, interactive evaluation, about, input reset, responsive navigation and confirmed deletion.
- The browser-created test observation was deleted through the actual confirmation UI. Existing observations
  were preserved. Clear-all is tested through confirmation/cancellation and backend tests on temporary data;
  existing user history was not erased for QA.

Evidence: [automated tests](evaluation/ui/tests.json), [full test log](evaluation/ui/tests.txt),
[browser result](evaluation/ui/browser_e2e.json), [artifact preservation](evaluation/ui/preservation.json),
[desktop screenshot](evaluation/ui/dashboard_desktop.png), [identification screenshot](evaluation/ui/identification_desktop.png),
[mobile screenshot](evaluation/ui/species_mobile.png), [tablet screenshot](evaluation/ui/dashboard_tablet.png).

## Run and verify

```powershell
Set-Location 'D:\\Haroon FYP\\AI-Weed-Assistant'
.\\.venv\\Scripts\\python.exe -m streamlit run app/main.py
```

Open **http://127.0.0.1:8501**. Colab is not required for predictions. To repeat checks:

```powershell
.\\.venv\\Scripts\\python.exe scripts/check_ui.py
node tools/browser-test/ui_e2e.cjs
```

The browser test requires the running local server and existing Microsoft Edge. If interrupted after saving
its own test record, `scripts/cleanup_ui_test_record.py` verifies the automated marker before removing it.
The original training-release browser script is historical; use `ui_e2e.cjs` for the redesigned routes.

## Remaining limits

- This is a verified local research application, not a publicly deployed multi-user production service.
- Only nine weed and three crop classes are trained. Independent Pakistani field accuracy, adult/mixed plants,
  unknown-species recognition and local agronomic suitability remain unverified.
- Camera UI is implemented, but headless QA does not establish physical camera/permission operation.
- Confidence is not proof of identity. Validation-only scaling increased held-out ECE; the original evaluation
  and model card retain that result. The threshold flags review and is not a proven open-set detector.
- Crop associations and chemical details remain unavailable where the curated sources do not verify them.
  No products/doses/mixtures are invented. User feedback never automatically updates weights or chatbot evidence.
- The workspace contained no separate proposal/Appendix A document; requirements were taken from the supplied
  mission and earlier pasted project requirements. Proposal-only features are not claimed implemented.

See [usage](docs/USAGE.md), [architecture](docs/ARCHITECTURE.md), [model card](MODEL_CARD.md) and
[limitations](docs/LIMITATIONS.md). No essential local UI feature remains blocked.
"""
    (ROOT/"UI_COMPLETION_REPORT.md").write_text(report,encoding="utf-8")
    print("Wrote UI_COMPLETION_REPORT.md from passing browser, test and unchanged-artifact evidence.")


if __name__=="__main__":main()
