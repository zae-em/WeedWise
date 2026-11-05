"""Bundle only training code/config; exclude data, environments, credentials and models."""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def prepare():
    target = ROOT / "training" / "colab_source.zip"
    required = ROOT / "requirements-colab.txt"
    if not required.is_file():
        raise FileNotFoundError("Finish environment verification and requirements generation first.")
    files = sorted((ROOT / "training").glob("*.py")) + [ROOT / "training/config.json", required]
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in files:
            archive.write(source, source.relative_to(ROOT).as_posix())
    print(f"Created {target}; {target.stat().st_size:,} bytes. No dataset included.")
    return target


if __name__ == "__main__":
    prepare()
