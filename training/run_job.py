"""Run a complete GPU job, record provenance and export a recoverable output archive."""
import argparse
import importlib.metadata
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
from .train import Config, train


def run(config_path):
    cfg=Config(**json.loads(Path(config_path).read_text(encoding="utf-8")))
    if cfg.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This full training job requires a CUDA GPU; no CPU fallback is allowed")
    torch.set_num_threads(2)
    output_root=Path(cfg.output_dir).parents[1]
    output_root.mkdir(parents=True,exist_ok=True)
    status_path=output_root/"job_status.json"
    started=time.time()
    provenance={"python":sys.version,"gpu":torch.cuda.get_device_name(0),"started_unix":started,
                "versions":{name:importlib.metadata.version(name) for name in ["torch","torchvision","numpy","scipy","scikit-learn","matplotlib","Pillow"]},
                "config":cfg.__dict__,"full_training_split":True,"cpu_fallback":False}
    Path(cfg.evaluation_dir).mkdir(parents=True,exist_ok=True)
    (Path(cfg.evaluation_dir)/"run_provenance.json").write_text(json.dumps(provenance,indent=2))
    (output_root/"runtime_requirements.txt").write_text(subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True))
    status_path.write_text(json.dumps({"state":"running","started_unix":started,"gpu":provenance["gpu"]},indent=2))
    try:
        train(cfg)
        artifact=torch.load(Path(cfg.output_dir)/"model.pt",map_location="cpu",weights_only=True)
        if not artifact.get("trained") or len(artifact["classes"]) < 2:
            raise ValueError("The exported artifact is not a completed trained classifier")
        status={"state":"complete","started_unix":started,"finished_unix":time.time(),"duration_seconds":time.time()-started,
                "gpu":provenance["gpu"],"model":str(Path(cfg.output_dir)/"model.pt"),"classes":artifact["classes"]}
        status_path.write_text(json.dumps(status,indent=2))
        destination=output_root.parent/"weed_artifacts"
        archive=shutil.make_archive(str(destination),"zip",output_root)
        print(json.dumps({**status,"output_archive":archive}),flush=True)
    except BaseException as error:
        status_path.write_text(json.dumps({"state":"error","error":str(error),"traceback":traceback.format_exc(),"duration_seconds":time.time()-started},indent=2))
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    run(parser.parse_args().config)
