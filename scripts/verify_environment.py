"""Verify installed dependencies, SQLite retrieval, notebooks and a CPU forward pass."""
import importlib
import importlib.metadata
import json
import os
import platform
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

PACKAGES = {
    "torch": "torch", "torchvision": "torchvision", "numpy": "numpy", "pandas": "pandas",
    "scikit-learn": "sklearn", "matplotlib": "matplotlib", "Pillow": "PIL", "opencv-python-headless": "cv2",
    "tqdm": "tqdm", "streamlit": "streamlit", "plotly": "plotly", "ImageHash": "imagehash",
    "albumentations": "albumentations", "jupyter": "jupyter_core", "ipykernel": "ipykernel", "notebook": "notebook",
}


def verify():
    results = {"checked_at": datetime.now().astimezone().isoformat(), "python": sys.version,
               "executable": sys.executable, "venv": sys.prefix != sys.base_prefix, "platform": platform.platform(), "packages": {}}
    errors = []
    for package, module in PACKAGES.items():
        try:
            importlib.import_module(module)
            results["packages"][package] = {"version": importlib.metadata.version(package), "import": "PASS"}
        except Exception as exc:
            errors.append(f"{package}: {exc}")
            results["packages"][package] = {"import": "FAIL", "error": str(exc)}
    try:
        import torch
        from training.train import build_model
        torch.set_num_threads(2)
        model = build_model(2, pretrained=False).eval()
        with torch.inference_mode():
            result = model(torch.zeros(1, 3, 224, 224))
        assert result.shape == (1, 2) and torch.isfinite(result).all()
        results["pytorch"] = {"cpu_forward": "PASS", "cuda_available": torch.cuda.is_available(), "cuda_build": torch.version.cuda,
                              "weights_downloaded": False, "training_run": False, "output_shape": list(result.shape)}
        assert torch.version.cuda is None, "Expected CPU-only local PyTorch"
    except Exception as exc:
        errors.append(f"torch CPU forward: {exc}")
    try:
        with sqlite3.connect(":memory:") as database:
            database.execute("CREATE VIRTUAL TABLE notes USING fts5(content)")
            database.execute("INSERT INTO notes VALUES (?)", ("local retrieval verification",))
            assert database.execute("SELECT count(*) FROM notes WHERE notes MATCH 'retrieval'").fetchone()[0] == 1
        from sklearn.feature_extraction.text import TfidfVectorizer
        assert TfidfVectorizer().fit_transform(["local retrieval", "text search"]).shape[0] == 2
        results["local_retrieval"] = {"sqlite_fts5": "PASS", "tfidf": "PASS", "sqlite_version": sqlite3.sqlite_version}
    except Exception as exc:
        errors.append(f"local retrieval: {exc}")
    try:
        import nbformat
        for notebook in (ROOT / "training").glob("*.ipynb"):
            nbformat.validate(nbformat.read(notebook, as_version=4))
        results["notebook_schema"] = "PASS"
    except Exception as exc:
        errors.append(f"notebook schema: {exc}")
    pip = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    results["pip_check"] = pip.stdout.strip() or pip.stderr.strip()
    if pip.returncode:
        errors.append("pip check failed")
    results["errors"] = errors
    results["status"] = "PASS" if not errors and results["venv"] else "FAIL"
    target = ROOT / "evaluation/setup_verification.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0 if results["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(verify())
