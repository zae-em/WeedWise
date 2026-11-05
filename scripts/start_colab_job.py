"""Queue supported VS Code notebook operations; no authentication secrets are accessed."""
import hashlib
import json
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def bootstrap_source(background=True):
    summary=json.loads((ROOT/"evaluation/dataset/summary.json").read_text())
    transfer_path=ROOT/"dataset/colab_parts/transfer_manifest.json"
    old_hash=json.loads(transfer_path.read_text())["sha256"] if transfer_path.exists() else summary["training_archive_sha256"]
    metadata_path=ROOT/"dataset/colab_parts/final_splits.json.gz"
    metadata_hash=hashlib.sha256(metadata_path.read_bytes()).hexdigest() if metadata_path.exists() else None
    with (ROOT/"training/colab_source.zip").open("rb") as stream:
        source_hash=hashlib.file_digest(stream,"sha256").hexdigest()
    code=f'''import hashlib, json, os, sys, subprocess, zipfile
from pathlib import Path
import torch
assert torch.cuda.is_available(), "A CUDA GPU is required; CPU fallback is disabled."
assert Path("/content").is_dir(), "Select the existing Colab GPU kernel."
source_zip=Path("/content/colab_source.zip")
dataset_zip=Path("/content/weed_training_224.zip")
if not dataset_zip.is_file():
    transfer=json.loads(Path("/content/transfer_manifest.json").read_text())
    temporary=dataset_zip.with_suffix(".joining")
    with temporary.open("wb") as outgoing:
        for part in transfer["parts"]:
            piece=Path("/content")/part["name"]
            assert piece.is_file(), f"Missing transfer part: {{piece.name}}"
            content=piece.read_bytes()
            assert hashlib.sha256(content).hexdigest()==part["sha256"], f"Transfer part hash mismatch: {{piece.name}}"
            outgoing.write(content)
    temporary.replace(dataset_zip)
for file, expected in [(source_zip,{source_hash!r})]:
    assert file.is_file(), f"Upload {{file.name}} with the official Colab extension."
    with file.open("rb") as stream:
        assert hashlib.file_digest(stream,"sha256").hexdigest()==expected, f"Transfer hash mismatch: {{file}}"
with dataset_zip.open("rb") as stream:
    image_archive_hash=hashlib.file_digest(stream,"sha256").hexdigest()
assert image_archive_hash in {{{summary['training_archive_sha256']!r},{old_hash!r}}}, "Dataset transfer SHA256 mismatch"
project=Path("/content/AI-Weed-Assistant")
project.mkdir(exist_ok=True)
with zipfile.ZipFile(source_zip) as archive:
    for entry in archive.infolist():
        assert (project/entry.filename).resolve().is_relative_to(project.resolve())
    archive.extractall(project)
os.chdir(project)
if str(project) not in sys.path: sys.path.insert(0,str(project))
from training.prepare_runtime import prepare_runtime_dataset
data=Path("/content/weed_dataset")
final_metadata=Path("/content/final_splits.json.gz")
if image_archive_hash!={summary['training_archive_sha256']!r}:
    assert final_metadata.is_file(), "Upload final_splits.json.gz to apply final group-aware assignments"
    assert hashlib.sha256(final_metadata.read_bytes()).hexdigest()=={metadata_hash!r}, "Final split manifest hash mismatch"
    manifest=prepare_runtime_dataset(dataset_zip,data,final_metadata)
else:
    manifest=prepare_runtime_dataset(dataset_zip,data)
assert len(manifest["classes"])=={summary['class_count']} and len(manifest["records"])=={summary['processed_images']}, "Dataset coverage mismatch"
for package in ["torchvision","numpy","scipy","sklearn","matplotlib","PIL"]:
    __import__(package)
cfg=json.loads(Path("training/config.json").read_text())
cfg.update(data_dir=str(data),output_dir="/content/weed_outputs/models/mobilenet_v2",evaluation_dir="/content/weed_outputs/evaluation/mobilenet_v2",device="cuda",num_workers=2)
resume=Path(cfg["output_dir"])/"last.pt"
uploaded_resume=Path("/content/last.pt")
if uploaded_resume.is_file() and not resume.exists():
    import shutil
    resume.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(uploaded_resume,resume)
if resume.is_file(): cfg["resume"]=str(resume)
Path("training/colab_config.json").write_text(json.dumps(cfg,indent=2))
print(json.dumps({{"gpu":torch.cuda.get_device_name(0),"classes":manifest["classes"],"full_split_counts":{{name:sum(r["split"]==name for r in manifest["records"]) for name in ["train","val","test"]}},"config":cfg}},indent=2))
'''
    if background:
        code+='''pid_file=Path("/content/weed_training_pid.json")
existing=None
if pid_file.exists():
    previous=json.loads(pid_file.read_text())
    try:
        os.kill(previous["pid"],0)
        command=Path(f"/proc/{previous['pid']}/cmdline").read_bytes()
        if b"training.run_job" in command: existing=previous["pid"]
    except ProcessLookupError:
        pass
if existing:
    print("Existing training process is still running:",existing)
else:
    outputs=Path("/content/weed_outputs");outputs.mkdir(exist_ok=True)
    log=(outputs/"training.log").open("a",buffering=1)
    process=subprocess.Popen([sys.executable,"-u","-m","training.run_job","--config","training/colab_config.json"],cwd=project,stdout=log,stderr=subprocess.STDOUT)
    pid_file.write_text(json.dumps({"pid":process.pid,"config":cfg},indent=2))
    print("Full-dataset GPU training process started:",process.pid)
'''
    else:
        code+='''subprocess.run([sys.executable,"-u","-m","training.run_job","--config","training/colab_config.json"],check=True)
'''
    return code


def write_notebook():
    notebook={"nbformat":4,"nbformat_minor":5,"metadata":{"colab":{"name":"complete_training.ipynb","provenance":[]},"accelerator":"GPU","kernelspec":{"name":"python3","display_name":"Python 3","language":"python"}},
              "cells":[{"cell_type":"markdown","id":"instructions","metadata":{},"source":["# Complete MobileNetV2 training\n","Select your Colab GPU kernel and Run All. Upload the prepared source bundle and 224px dataset ZIP first using the official extension. This trains on the complete training split, calibrates on validation only, evaluates the held-out test split, and exports outputs. No Google Drive is used.\n"]},
                       {"cell_type":"code","id":"full-training","metadata":{},"execution_count":None,"outputs":[],"source":bootstrap_source(False).splitlines(True)}]}
    (ROOT/"training/complete_training.ipynb").write_text(json.dumps(notebook,indent=2),encoding="utf-8")
    return ROOT/"training/complete_training.ipynb"


def queue():
    request={"id":"fulltrain-"+str(time.time_ns()),"action":"appendExecute","notebook":"training/colab_connection_test.ipynb","timeoutSeconds":900,"source":bootstrap_source(True)}
    (ROOT/".weed-control/request.json").write_text(json.dumps(request),encoding="utf-8")
    write_notebook()
    print("Queued "+request["id"]+" using the selected Colab kernel; wrote reproducible complete_training.ipynb.")


if __name__=="__main__":queue()
