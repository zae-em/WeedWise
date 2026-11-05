"""Monitor the authorized Colab job through the project VS Code controller and retrieve its outputs."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CONTROL=ROOT/".weed-control"


def request(action,timeout=240,**kwargs):
    prior=CONTROL/"request.json"
    deadline=time.monotonic()+timeout
    while prior.exists():
        previous=json.loads(prior.read_text(encoding="utf-8"))
        result_path=CONTROL/(previous["id"]+".json")
        if result_path.exists() and json.loads(result_path.read_text()).get("status")!="running":break
        if time.monotonic()>deadline:raise TimeoutError("Previous VS Code action is still pending")
        time.sleep(1)
    identifier="watch-"+str(time.time_ns())
    payload={"id":identifier,"action":action,**kwargs}
    temporary=CONTROL/"request.pending"
    temporary.write_text(json.dumps(payload),encoding="utf-8");temporary.replace(prior)
    result_path=CONTROL/(identifier+".json")
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if result_path.exists():
            try:result=json.loads(result_path.read_text())
            except json.JSONDecodeError:time.sleep(.5);continue
            if result.get("status")=="complete":return result
            if result.get("status") in {"error","interrupted"}:raise RuntimeError(result.get("error") or result.get("note"))
        time.sleep(.5)
    raise TimeoutError("VS Code action did not finish; check Colab connection")


def main():
    last_epoch=None
    errors=0
    print("Watching the full-dataset GPU job. Credentials remain managed by the official VS Code extension.",flush=True)
    while True:
        try:
            request("readRemote",remotePath="/content/weed_outputs/job_status.json",destination="evaluation/training_job.json")
            job=json.loads((ROOT/"evaluation/training_job.json").read_text())
            if job["state"]=="error":
                print("TRAINING ERROR: "+json.dumps(job),flush=True);return 1
            request("readRemote",remotePath="/content/weed_outputs/models/mobilenet_v2/training_status.json",destination="evaluation/training_progress.json")
            progress=json.loads((ROOT/"evaluation/training_progress.json").read_text())
            epoch=progress.get("completed_epoch",progress.get("completed_epochs"))
            if epoch!=last_epoch:
                print(json.dumps(progress),flush=True);last_epoch=epoch
            if job["state"]=="complete":
                print("Training complete; retrieving the model, evaluation and checkpoints to the laptop.",flush=True)
                result=request("readRemote",timeout=900,remotePath="/content/weed_artifacts.zip",destination="models/weed_artifacts.zip")
                print("Downloaded artifact archive: "+json.dumps(result),flush=True)
                subprocess.run([sys.executable,"scripts/import_artifacts.py",str(ROOT/"models/weed_artifacts.zip")],cwd=ROOT,check=True)
                print("LOCAL_ARTIFACTS_READY",flush=True)
                return 0
            errors=0
        except Exception as error:
            errors+=1
            print(f"Monitor attempt {errors}: {type(error).__name__}: {error}",flush=True)
            if errors>=8:
                print("Monitoring needs intervention; existing local checkpoints remain preserved.",flush=True);return 1
        time.sleep(30)


if __name__=="__main__":raise SystemExit(main())
