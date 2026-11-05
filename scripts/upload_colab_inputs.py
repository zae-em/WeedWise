"""Prepare small hashed inputs and queue uploads through the authenticated official VS Code provider."""
import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_colab_bundle import prepare
from scripts.start_colab_job import write_notebook


def main(resume=None):
    prepare()
    source=ROOT/"dataset/weed_training_224.zip"
    if not source.is_file():raise FileNotFoundError("Prepare the provided dataset before uploading")
    folder=ROOT/"dataset/colab_parts";folder.mkdir(exist_ok=True)
    entries=[]
    with source.open("rb") as stream:
        index=0
        while content:=stream.read(8*1024*1024):
            path=folder/f"weed_training_224.zip.part{index:03d}"
            digest=hashlib.sha256(content).hexdigest()
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:path.write_bytes(content)
            entries.append({"name":path.name,"bytes":len(content),"sha256":digest});index+=1
    with source.open("rb") as stream:digest=hashlib.file_digest(stream,"sha256").hexdigest()
    (folder/"transfer_manifest.json").write_text(json.dumps({"archive_name":source.name,"sha256":digest,"parts":entries},indent=2))
    payload={"manifest":json.loads((ROOT/"dataset/processed/manifest.json").read_text()),"summary":json.loads((ROOT/"evaluation/dataset/summary.json").read_text())}
    (folder/"final_splits.json.gz").write_bytes(gzip.compress(json.dumps(payload).encode("utf-8"),mtime=0))
    files=["training/colab_source.zip","dataset/colab_parts/transfer_manifest.json","dataset/colab_parts/final_splits.json.gz"]+["dataset/colab_parts/"+entry["name"] for entry in entries]
    if resume:
        candidate=Path(resume).resolve()
        if not candidate.is_relative_to(ROOT):raise ValueError("Resume checkpoint must remain inside this project")
        if candidate.name!="last.pt":raise ValueError("Copy the recovery checkpoint into a project recovery folder with the filename last.pt first")
        files.append(candidate.relative_to(ROOT).as_posix())
        print("The notebook will restore the uploaded last.pt and verify its dataset and configuration before resuming.")
    write_notebook()
    control=ROOT/".weed-control"
    ready=control/"ready.json"
    if not ready.is_file():raise RuntimeError("Open this project in VS Code with the project notebook controller enabled")
    identifier="upload-inputs-"+str(time.time_ns())
    request={"id":identifier,"action":"upload","files":files}
    current=control/"request.json"
    if current.exists():
        prior=json.loads(current.read_text())
        result=control/(prior["id"]+".json")
        if not result.exists() or json.loads(result.read_text()).get("status")=="running":raise RuntimeError("A prior controller action is still active; wait for it to finish")
    current.write_text(json.dumps(request),encoding="utf-8")
    print(f"Queued {len(files)} inputs as {identifier}. Choose the notebook's existing T4 server if prompted.",flush=True)
    deadline=time.monotonic()+1800
    result=control/(identifier+".json")
    while time.monotonic()<deadline:
        if result.exists():
            try:state=json.loads(result.read_text())
            except json.JSONDecodeError:time.sleep(1);continue
            if state.get("status")=="complete":
                print("Upload command returned. The complete notebook verifies all part and archive hashes before training.")
                return
            if state.get("status") in {"error","interrupted"}:raise RuntimeError(state.get("error") or state.get("note"))
        time.sleep(1)
    raise TimeoutError("Upload has not finished; inspect VS Code's Colab output and selection prompts")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--resume",help="Optional project-local recovery checkpoint")
    main(parser.parse_args().resume)
