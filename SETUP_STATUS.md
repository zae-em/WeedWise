# Historical setup status

This is the initial environment-setup snapshot, retained for its version and hardware evidence.
Its dataset, training, application and agent-execution rows describe the earlier setup stage and are
superseded by [FINAL_PROJECT_STATUS.md](FINAL_PROJECT_STATUS.md). Full T4 training, model retrieval and
application integration have since completed through supported VS Code notebook and Colab filesystem APIs.

Prepared 31 October 2026 (Asia/Karachi).

Project: `D:\Haroon FYP\AI-Weed-Assistant`

## Verification

| Item | Result | Evidence / scope |
| --- | --- | --- |
| Python / pip | PASS | Python 3.12 project environment; executable: `D:\Haroon FYP\AI-Weed-Assistant\.venv\Scripts\python.exe` |
| Virtual environment | PASS | Project `.venv`; VS Code interpreter setting and local notebook kernel configured |
| Required libraries | PASS | All requested imports and `pip check`; exact versions below |
| VS Code extensions | PASS | Four requested extensions verified by VS Code CLI; dependencies installed automatically |
| Local PyTorch CPU | PASS | CPU MobileNetV2 forward pass with random weights; no pretrained weights downloaded and no optimization |
| Local notebook kernel | PASS | Executed a CPU diagnostic in the project .venv kernel and checked its Python executable |
| Google authentication | VERIFIED indirectly by successful saved cloud cell execution | Verified only if saved remote execution evidence exists; no OAuth tokens inspected or stored in project |
| Colab GPU | VERIFIED | Actual reported GPU: **Tesla T4**; CUDA tensor test: True |
| Cloud notebook execution | VERIFIED through VS Code notebook output (user executed) | The user executed the cell through VS Code; this agent cannot directly execute the connected kernel using its available tools |
| Local folders | PASS | `dataset/`, `training/`, `models/`, `evaluation/`, `knowledge_base/`, `app/`, `tests/` |
| Training infrastructure | PASS | Configurable MobileNetV2 pipeline, two notebooks, transfer helpers and synthetic infrastructure tests |
| Dataset staging | READY | Place your ZIP in `dataset/` later. No dataset was downloaded, moved, inspected or extracted |
| Actual model training | NOT STARTED | No dataset statistics, species accuracy or trained model exists; notebook START_TRAINING remains False |
| Streamlit application | DEFERRED AS REQUESTED | Dependencies and app folder only; full application development has not begun |

## Existing hardware and software

- OS: Microsoft Windows 11 Pro 10.0.26200, 64-bit.
- CPU: Intel(R) Core(TM) i5-8365U CPU @ 1.60GHz, 4 cores / 8 logical processors.
- RAM: 15.8 GiB visible.
- Graphics: Intel(R) UHD Graphics 620; no NVIDIA GPU detected.
- Reused working Python 3.12.10, Git 2.55.0.windows.3 and VS Code 1.144.0; no reinstallation.
- Python 3.14 also exists on the laptop; this project deliberately uses Python 3.12.
- Initial free space: approximately 32 GiB on C: and 448 GiB on D:; project lives on D:.
- Existing Codex extension was already installed. Windows security settings and execution policy were not changed.

## Installed requested extensions

- `ms-python.python@2026.10.0`
- `ms-python.vscode-pylance@2026.4.1`
- `ms-toolsai.jupyter@2025.9.1`
- `google.colab@0.9.7`

Additional Python/Jupyter dependency extensions were installed by VS Code automatically.

## Python dependency verification

| Package | Installed version | Import |
| --- | --- | --- |
| torch | 2.15.0+cpu | PASS |
| torchvision | 0.29.2+cpu | PASS |
| numpy | 2.5.4 | PASS |
| pandas | 3.1.0 | PASS |
| scikit-learn | 1.9.1 | PASS |
| matplotlib | 3.11.2 | PASS |
| Pillow | 13.0.0 | PASS |
| opencv-python-headless | 5.0.0.93 | PASS |
| tqdm | 4.70.1 | PASS |
| streamlit | 1.66.1 | PASS |
| plotly | 7.1.0 | PASS |
| ImageHash | 4.3.2 | PASS |
| albumentations | 2.0.8 | PASS |
| jupyter | 1.1.1 | PASS |
| ipykernel | 7.4.0 | PASS |
| notebook | 7.6.4 | PASS |

Dependency consistency: `No broken requirements found.`.
OpenCV uses opencv-python-headless (`cv2`) to avoid duplicate OpenCV builds. SQLite is included in Python.
SQLite FTS5 and TF-IDF retrieval are verified in memory; no external AI service or persistent database was created.
`requirements.txt` pins the actual Windows environment, including transitive packages and CPU PyTorch.
`requirements-colab.txt` pins portable training dependencies and leaves Colab's matched CUDA torch/torchvision in place.
The Colab training dependency installation is prepared but has not been executed remotely.
Initial imports of Pandas/SciPy were blocked by Windows Application Control (CodeIntegrity event 3077). A fresh Python process subsequently passed every import and the full environment check using the same installed versions. No security setting, policy or package downgrade was applied.
Package download timeouts were resolved using resumable official PyPI wheel downloads with SHA-256 validation, followed by installation from the local cache.

## Successful tests

- Infrastructure suite: 17 tests, 0 failures, 0 errors.
- Tests use temporary synthetic images and mocked epochs; no real model training or dataset access.
- Coverage includes split reproducibility, duplicate leakage, augmentations, unsafe ZIP rejection, disk capacity, CPU inference, top-k counting, random state serialization, phase changes, early stopping and resume.
- Local CPU inference uses an untrained MobileNetV2 and validates tensor shape and finite values; this measures no species performance.
- Notebook schemas: PASS; local notebook execution: PASS. These are separate from the saved Colab GPU result.

## Colab evidence and limits

- Authentication: VERIFIED indirectly by successful saved cloud cell execution.
- GPU: Tesla T4; CUDA available: True; GPU tensor test: True.
- Remote PyTorch: `2.11.0+cu130`; CUDA build: `13.0`.
- Remote Python: `3.13.16 (main, Oct  1 2026, 15:40:24) [GCC 13.3.0]`; working directory: `/content`.
- The initial Colab assignment error about too many servers was resolved through user intervention before the saved GPU test.
- A saved successful cell verifies cloud execution at that time; it does not guarantee a server is still assigned later.
- Programmatic agent control of the currently connected Colab kernel is unavailable in this session. The official extension exposes UI commands; installing it does not grant this assistant an authenticated notebook execution tool.
- No Drive mount, cloud storage account, permanent cloud dataset, remote tunnel or extracted OAuth credential was configured.

## Ready for later / manual workflow

1. Put your dataset ZIP in `dataset/` when ready. No setup step waits for it.
2. Use the official VS Code Upload to Colab command for `training/colab_source.zip` and, later, your dataset ZIP.
3. Follow `training/train_colab.ipynb`: verify GPU, unpack source, install portable dependencies, safely extract your uploaded ZIP, set the actual class-folder root and inspect detected classes.
4. Enable START_TRAINING only when you choose to begin. No training has run during setup.
5. Download last.pt periodically through Colab Contents to local models/. It includes the best weights, optimizer, early stopping and RNG state. Runtime-only files can disappear after termination.
6. Download final weed_artifacts.zip and run scripts/import_artifacts.py to restore models/ and evaluation/ locally. Existing output files are protected from overwriting.
7. Reconnect/re-authenticate or choose an available GPU if Colab expires or reaches quota. GPU type and availability remain subject to Google allocation.

## Evidence files

- `evaluation/setup_verification.json`: imports, versions, CPU forward, SQLite retrieval and pip check.
- `evaluation/setup_tests.json` and `evaluation/setup_tests.txt`: synthetic test results.
- `evaluation/setup_colab_status.json` and saved `training/colab_connection_test.ipynb`: remote GPU evidence.
- `evaluation/setup_local_notebook.json`: local kernel smoke execution.
- `evaluation/setup_extensions.txt`: installed extension versions.
- `COLAB_WORKFLOW.md`: exact transfer, interruption and recovery steps.

## Official references

- [Colab extension guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide)
- [Colab extension commands](https://github.com/googlecolab/colab-vscode/blob/main/package.json)
- [Colab resource limits](https://research.google.com/colaboratory/faq.html#resource-limits)
- [PyTorch local installation](https://pytorch.org/get-started/locally/)
