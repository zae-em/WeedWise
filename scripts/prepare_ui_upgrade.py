"""Preserve UI/database state and fingerprint protected ML artifacts before the UI upgrade."""
import hashlib
import json
import shutil
import sqlite3
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def protected_hashes():
    files=list((ROOT/"models").rglob("*.pt"))+list((ROOT/"models").rglob("*.pth"))
    files += [p for p in (ROOT/"evaluation/mobilenet_v2").rglob("*") if p.is_file()]
    files += [ROOT/"dataset/processed/manifest.json",ROOT/"knowledge_base/species.json"]
    result={}
    for path in sorted(files):
        with path.open("rb") as stream:result[path.relative_to(ROOT).as_posix()]=hashlib.file_digest(stream,"sha256").hexdigest()
    return result


def main():
    output=ROOT/"evaluation/ui";output.mkdir(exist_ok=True)
    baseline=output/"protected_artifacts_before.json"
    if baseline.exists():
        assert json.loads(baseline.read_text())==protected_hashes(),"A protected artifact differs from the original UI baseline"
        print("Existing protected-artifact baseline verified; backup preserved.")
        return
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup=ROOT/".ui-backup"/stamp
    backup.mkdir(parents=True)
    for folder in ["app","weed_assistant","tests"]:
        for source in (ROOT/folder).rglob("*"):
            if not source.is_file() or "__pycache__" in source.parts or "uploads" in source.parts or "data" in source.parts:continue
            target=backup/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    db=ROOT/"app/data/weed_assistant.sqlite3"
    if db.exists():
        with sqlite3.connect(db) as source,sqlite3.connect(backup/"weed_assistant.sqlite3") as target:source.backup(target)
    baseline.write_text(json.dumps(protected_hashes(),indent=2),encoding="utf-8")
    (output/"backup.json").write_text(json.dumps({"directory":str(backup),"created_utc":stamp,"sqlite_backup":db.exists()},indent=2))
    print("Backed up working application and SQLite to",backup)
    print("Recorded unchanged-artifact baseline for model weights, evaluation, split and knowledge.")


if __name__=="__main__":main()
