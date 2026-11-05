"""Regenerate clean, local notebooks; never connect to or train on a runtime."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write(name, cells):
    if (ROOT / "training" / name).exists():
        print(f"Preserving existing notebook and its outputs: {name}")
        return
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "colab": {"name": name, "provenance": []},
            "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "cells": [],
    }
    for i, (kind, source) in enumerate(cells):
        cell = {"cell_type": kind, "id": f"weed-{i:03}", "metadata": {}, "source": source.splitlines(True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        notebook["cells"].append(cell)
    (ROOT / "training" / name).write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")


GPU_TEST = '''import json, platform, sys
from pathlib import Path
import torch
available = torch.cuda.is_available()
result = {
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "cuda_build": torch.version.cuda,
    "cuda_available": available,
    "gpu_name": torch.cuda.get_device_name(0) if available else None,
    "execution_location": str(Path.cwd()),
    "is_colab_runtime": Path("/content").is_dir() and "google.colab" in sys.modules,
}
if available:
    x = torch.randn(64, 64, device="cuda")
    result["cuda_tensor_test"] = bool(torch.isfinite(x @ x.T).all().item())
else:
    result["cuda_tensor_test"] = False
print(json.dumps(result, indent=2))
if Path("/content").is_dir():
    Path("/content/colab_gpu_status.json").write_text(json.dumps(result, indent=2))
assert available, "No CUDA GPU: select a Colab GPU server, then run this cell again."
'''


if __name__ == "__main__":
    write("colab_connection_test.ipynb", [
        ("markdown", "# Colab GPU connection test\nSelect Kernel > Select Another Kernel > Colab > New Colab Server > GPU (a free option if offered). Sign in personally when prompted, then Run All and save this local notebook. Auto Connect may select CPU, so verify the output. This notebook only tests compute; it never trains or accesses your dataset. Notebook metadata alone does not allocate a GPU.\n"),
        ("code", GPU_TEST),
    ])
    write("train_colab.ipynb", [
        ("markdown", "# MobileNetV2 training (local storage, temporary Colab compute)\nKeep this notebook on your laptop. Select a Colab GPU kernel. Run preparation cells individually; training has a separate opt-in guard. Do not use Run All until you are ready to train. No Google Drive mount is used. See COLAB_WORKFLOW.md for upload, download and recovery.\n"),
        ("code", GPU_TEST),
        ("markdown", "## Transfer code\nOn the laptop run `scripts/prepare_colab_bundle.py`. In VS Code Explorer right-click `training/colab_source.zip` > Upload to Colab. Also upload your dataset ZIP later and any resume checkpoint. Uploads go to temporary `/content`.\n"),
        ("code", '''from pathlib import Path
import zipfile
source = Path("/content/colab_source.zip")
assert source.exists(), "Upload training/colab_source.zip using VS Code > Upload to Colab."
project = Path("/content/AI-Weed-Assistant")
with zipfile.ZipFile(source) as archive:
    # The bundle is generated locally from an explicit source allowlist.
    for entry in archive.infolist():
        dest = (project / entry.filename).resolve()
        if not dest.is_relative_to(project.resolve()):
            raise ValueError("Unsafe archive path")
    archive.extractall(project)
%cd /content/AI-Weed-Assistant
'''),
        ("markdown", "## Install runtime dependencies\nUse the runtime's CUDA PyTorch build. Never install the Windows CPU requirements.txt on Colab. This cell installs only portable training dependencies and records actual runtime versions.\n"),
        ("code", '''%pip install -r requirements-colab.txt
import importlib.metadata as metadata
import json
from pathlib import Path
versions = {p: metadata.version(p) for p in ["torch", "torchvision", "numpy", "scikit-learn", "matplotlib", "Pillow", "tqdm"]}
Path("runtime_versions.json").write_text(json.dumps(versions, indent=2))
print(versions)
'''),
        ("markdown", "## Extract the dataset when available\nSet the archive path after uploading. Extraction checks free disk space and rejects unsafe paths; it does not download anything. Point DATA_DIR at the actual class-folder root (or root containing train/val/test) after extraction.\n"),
        ("code", '''from training.archive import extract_dataset
DATASET_ZIP = Path("/content/CHANGE_ME.zip")
EXTRACT_TO = Path("/content/weed_dataset")
assert DATASET_ZIP.is_file(), "Upload the dataset ZIP and set DATASET_ZIP."
extract_dataset(DATASET_ZIP, EXTRACT_TO)
print("Extracted top-level entries:", [p.name for p in EXTRACT_TO.iterdir()])
'''),
        ("code", '''import json
DATA_DIR = EXTRACT_TO  # Adjust if the ZIP contains an enclosing directory.
config = json.loads(Path("training/config.json").read_text())
config.update(data_dir=str(DATA_DIR), output_dir="/content/weed_outputs/models", evaluation_dir="/content/weed_outputs/evaluation", device="cuda", num_workers=2)
# To resume, upload the self-contained last.pt and set config["resume"] below.
# config["resume"] = "/content/resume/last.pt"
Path("training/colab_config.json").write_text(json.dumps(config, indent=2))
from training.train import inspect_data
print(inspect_data(config))  # Validates folders and records splits, no training.
'''),
        ("markdown", "## Explicit training step\nSet START_TRAINING to True only after the real dataset is ready and inspected. The command automatically uses ImageNet transfer learning, then fine-tunes. The first training invocation downloads the official MobileNetV2 weights if absent.\n"),
        ("code", '''START_TRAINING = False
if START_TRAINING:
    from training.train import Config, train
    train(Config(**json.loads(Path("training/colab_config.json").read_text())))
else:
    print("Training is disabled. No model training has run.")
'''),
        ("markdown", "## Save outputs to the laptop\nAfter each useful epoch, use the Colab Contents view to download `/content/weed_outputs/models/last.pt` and `best.pt` into local `models/` (or package below after training stops). Downloads are manual; runtime files alone cannot survive runtime deletion. Download a new snapshot periodically. To resume a new runtime, re-upload source, dataset, last.pt and best.pt, then set resume in the config.\n"),
        ("code", '''import shutil
from pathlib import Path
outputs = Path("/content/weed_outputs")
assert outputs.exists(), "No training outputs exist yet."
shutil.copy2("runtime_versions.json", outputs / "runtime_versions.json")
archive = shutil.make_archive("/content/weed_artifacts", "zip", outputs)
print("VS Code: Colab Contents > weed_artifacts.zip > Download...; save locally.")
print("Then run: .venv\\\\Scripts\\\\python.exe scripts/import_artifacts.py <downloaded-zip>")
# Browser Colab fallback only: from google.colab import files; files.download(archive)
'''),
    ])
    print("Created both notebooks without executing them.")
