"""Run the complete existing and redesigned UI test suite without overwriting ML evaluation reports."""
import io
import json
import sys
import unittest
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
if __name__=="__main__":
    stream=io.StringIO();suite=unittest.defaultTestLoader.discover(str(ROOT/"tests"))
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    directory=ROOT/"evaluation/ui";directory.mkdir(exist_ok=True)
    (directory/"tests.txt").write_text(stream.getvalue(),encoding="utf-8")
    summary={"finished_utc":datetime.now(timezone.utc).isoformat(),"tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"skipped":[str(t) for t,_ in result.skipped],"passed":result.wasSuccessful(),"retrained":False}
    (directory/"tests.json").write_text(json.dumps(summary,indent=2));print(stream.getvalue());print(json.dumps(summary,indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
