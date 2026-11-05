"""Pin the actual installed Windows environment; keep CUDA PyTorch out of Colab pins."""
import importlib.metadata
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    header = "# Verified Windows x64 / Python 3.12 environment. Do not use this file on Colab.\n--extra-index-url https://download.pytorch.org/whl/cpu\n"
    (ROOT / "requirements.txt").write_text(header + frozen, encoding="utf-8")
    # The Colab runtime supplies the matched CUDA torch/torchvision pair.
    portable = ["numpy", "scikit-learn", "matplotlib", "Pillow", "tqdm"]
    lines = ["# Colab only. Preserve the runtime's CUDA-enabled torch + torchvision."]
    lines += [f"{name}=={importlib.metadata.version(name)}" for name in portable]
    (ROOT / "requirements-colab.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Pinned local requirements.txt and portable requirements-colab.txt")
