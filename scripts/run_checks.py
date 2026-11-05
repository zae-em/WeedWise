"""Run infrastructure, data, application and available real-model tests; save their actual result."""
import contextlib
import io
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    text = stream.getvalue()
    (ROOT / "evaluation/setup_tests.txt").write_text(text, encoding="utf-8")
    summary = {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
               "passed": result.wasSuccessful(), "real_training_performed_by_test_suite": False,
               "real_model_available": (ROOT / "models/mobilenet_v2/model.pt").is_file(),
               "skipped": [{"test":test.id(),"reason":reason} for test,reason in result.skipped],
               "dataset_used": True, "scope": "Infrastructure, full processed-dataset validation, SQLite, grounded retrieval, Streamlit navigation, and genuine-model inference when available"}
    (ROOT / "evaluation/setup_tests.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(text)
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
