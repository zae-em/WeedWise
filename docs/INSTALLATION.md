# Installation and environment

The Windows setup uses Python 3.12.10 in `.venv`; VS Code points to `.venv/Scripts/python.exe`.
Python, Pylance, Jupyter and the official Google Colab extension are installed. Git and VS Code were reused.
CPU PyTorch is used locally; Colab retains its own matched CUDA PyTorch/torchvision builds.

From the repository root, create the Windows x64 environment with Python 3.12:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/verify_environment.py
```

`requirements.txt` pins the actual verified environment, including transitive packages. Its CPU PyTorch index
is intentional. Do not install this Windows file into Colab. Actual GPU run versions are recorded with the model.
OpenCV is opencv-python-headless (`cv2`); SQLite is built into Python. Local TF-IDF retrieval requires no LLM API.
Application fonts use system fonts, and no external font service is needed.

Browser tests use Node.js and the existing Microsoft Edge installation. Dependencies are pinned in
`tools/browser-test/package.json` and package-lock.json. To reconstruct them:

```powershell
npm.cmd ci --prefix tools/browser-test --no-audit --no-fund
```

No separate browser download is required when the Edge channel is installed. Select the project `.venv` for
local Python development, **AI Weed Assistant (.venv CPU)** for local notebooks, and a Colab GPU kernel for training.
Calling the Python executable directly avoids changing PowerShell execution policy.

Windows Application Control initially blocked two native modules during setup. A fresh process passed the full
import check with the same versions. Security policies and protections were not changed.
