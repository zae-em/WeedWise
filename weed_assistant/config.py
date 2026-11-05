from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
ACTIVE_VERSION="v1"
RELEASE={}
if (ROOT/"models/active.json").is_file():
    RELEASE=json.loads((ROOT/"models/active.json").read_text())
    ACTIVE_VERSION=RELEASE["version"]
    if ACTIVE_VERSION not in {"v1","v2"}:raise ValueError("Unknown active model version")
MODEL_PATH=ROOT/("models/v2/model.pt" if ACTIVE_VERSION=="v2" else "models/mobilenet_v2/model.pt")
DATABASE_PATH=ROOT/"app/data/weed_assistant.sqlite3"
UPLOADS_PATH=ROOT/"app/uploads"
KNOWLEDGE_PATH=ROOT/("knowledge_base/species_v2.json" if ACTIVE_VERSION=="v2" else "knowledge_base/species.json")
EVALUATION_PATH=ROOT/("evaluation/v2" if ACTIVE_VERSION=="v2" else "evaluation/mobilenet_v2")
DATASET_SUMMARY_PATH=ROOT/("dataset/combined_v2/summary.json" if ACTIVE_VERSION=="v2" else "evaluation/dataset/summary.json")
CROP_CLASSES={name for name,role in RELEASE.get("class_roles",{}).items() if role=="crop"} if ACTIVE_VERSION=="v2" else {"Common wheat","Maize","Sugar beet"}
MAX_IMAGE_BYTES=12*1024*1024
