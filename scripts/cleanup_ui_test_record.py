"""Remove only a failed browser run's own explicitly marked verification record."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from weed_assistant.database import Database
from weed_assistant.config import UPLOADS_PATH

if __name__=="__main__":
    path=ROOT/"evaluation/ui/browser_e2e.json"
    if not path.exists():raise SystemExit(0)
    report=json.loads(path.read_text());identifier=report.get("createdTestRecord")
    if report.get("status")=="FAIL" and identifier:
        db=Database()
        if not db.get_prediction(identifier):raise SystemExit(0)
        notes=[row for row in db.feedback() if row["prediction_id"]==identifier]
        if not notes or not all(row["comment"].startswith("Automated UI verification") for row in notes):
            raise RuntimeError("Refusing to remove a record without the automated-test marker")
        print("Removed only the failed browser test record:",db.delete_predictions([identifier],UPLOADS_PATH))
