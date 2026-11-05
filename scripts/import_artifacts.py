"""Import a downloaded Colab output bundle into local models/ and evaluation/."""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.archive import extract_dataset


def import_bundle(archive):
    archive = Path(archive).resolve()
    with tempfile.TemporaryDirectory(prefix="weed-import-", dir=ROOT / "models") as temporary:
        extracted = extract_dataset(archive, Path(temporary) / "extracted", reserve_bytes=100*1024**2)
        # Allow only output folders created by the notebook, with collision protection.
        planned = []
        for folder in ["models", "evaluation"]:
            source = extracted / folder
            if source.is_dir():
                for file in source.rglob("*"):
                    if file.is_file():
                        destination = ROOT / folder / file.relative_to(source)
                        if destination.exists():
                            raise FileExistsError(f"Already exists: {destination}. Archive the previous run locally first.")
                        planned.append((file, destination))
        extras = {"job_status.json": ROOT / "evaluation/mobilenet_v2/job_status.json",
                  "runtime_requirements.txt": ROOT / "models/mobilenet_v2/runtime_requirements.txt",
                  "training.log": ROOT / "evaluation/mobilenet_v2/training.log"}
        for name, destination in extras.items():
            source = extracted / name
            if source.is_file():
                if destination.exists():
                    raise FileExistsError(f"Already exists: {destination}")
                planned.append((source, destination))
        if not planned:
            raise ValueError("No models/ or evaluation/ files found in artifact bundle")
        for source, destination in planned:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        runtime = extracted / "runtime_versions.json"
        if runtime.is_file():
            destination = ROOT / "models" / "runtime_versions.json"
            if not destination.exists():
                shutil.copy2(runtime, destination)
    print(f"Imported {len(planned)} output files to {ROOT}; checkpoints were not executed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", help="Downloaded weed_artifacts.zip path")
    import_bundle(parser.parse_args().archive)
