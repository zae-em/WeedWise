"""Build the setup report from saved verification evidence, without accessing secrets."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(name):
    path = ROOT / "evaluation" / name
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}


def report():
    local = read_json("setup_verification.json")
    tests = read_json("setup_tests.json")
    local_notebook = read_json("setup_local_notebook.json")
    colab = read_json("setup_colab_status.json")
    os_info = read_json("setup_os.json")
    cpu = read_json("setup_cpu.json")
    graphics = read_json("setup_graphics.json")
    extensions_path = ROOT / "evaluation/setup_extensions.txt"
    extensions = extensions_path.read_text(encoding="utf-8-sig").splitlines() if extensions_path.exists() else []
    needed = ["ms-python.python", "ms-python.vscode-pylance", "ms-toolsai.jupyter", "google.colab"]
    installed = {name: next((item for item in extensions if item.startswith(name + "@")), "NOT FOUND") for name in needed}
    required_folders = ["dataset", "training", "models", "evaluation", "knowledge_base", "app", "tests"]
    all_folders = all((ROOT / name).is_dir() for name in required_folders)
    local_pass = local.get("status") == "PASS"
    passed = "PASS" if local_pass else "NOT YET VERIFIED"
    cloud = colab.get("test_output", {})
    rows = [
        ("Python / pip", passed, f"Python 3.12 project environment; executable: `{local.get('executable', 'pending')}`"),
        ("Virtual environment", "PASS" if local.get("venv") else "NOT VERIFIED", "Project `.venv`; VS Code interpreter setting and local notebook kernel configured"),
        ("Required libraries", passed, "All requested imports and `pip check`; exact versions below"),
        ("VS Code extensions", "PASS" if all(v != "NOT FOUND" for v in installed.values()) else "INCOMPLETE", "Four requested extensions verified by VS Code CLI; dependencies installed automatically"),
        ("Local PyTorch CPU", local.get("pytorch", {}).get("cpu_forward", "NOT VERIFIED"), "CPU MobileNetV2 forward pass with random weights; no pretrained weights downloaded and no optimization"),
        ("Local notebook kernel", local_notebook.get("status", "NOT YET VERIFIED"), "Executed a CPU diagnostic in the project .venv kernel and checked its Python executable"),
        ("Google authentication", colab.get("authentication", "NOT VERIFIED"), "Verified only if saved remote execution evidence exists; no OAuth tokens inspected or stored in project"),
        ("Colab GPU", colab.get("gpu_connection", "NOT VERIFIED"), f"Actual reported GPU: **{colab.get('gpu_name') or 'unknown'}**; CUDA tensor test: {cloud.get('cuda_tensor_test', 'not run')}"),
        ("Cloud notebook execution", colab.get("cloud_cell_execution", "NOT VERIFIED"), "The user executed the cell through VS Code; this agent cannot directly execute the connected kernel using its available tools"),
        ("Local folders", "PASS" if all_folders else "INCOMPLETE", ", ".join(f"`{name}/`" for name in required_folders)),
        ("Training infrastructure", "PASS" if tests.get("passed") else "NOT YET VERIFIED", "Configurable MobileNetV2 pipeline, two notebooks, transfer helpers and synthetic infrastructure tests"),
        ("Dataset staging", "READY", "Place your ZIP in `dataset/` later. No dataset was downloaded, moved, inspected or extracted"),
        ("Actual model training", "NOT STARTED", "No dataset statistics, species accuracy or trained model exists; notebook START_TRAINING remains False"),
        ("Streamlit application", "DEFERRED AS REQUESTED", "Dependencies and app folder only; full application development has not begun"),
    ]
    lines = ["# Setup status", "", "Prepared 31 October 2026 (Asia/Karachi).", "",
             f"Project: `{ROOT}`", "", "## Verification", "", "| Item | Result | Evidence / scope |", "| --- | --- | --- |"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    lines += ["", "## Existing hardware and software", "",
              f"- OS: {os_info.get('Caption', 'unknown')} {os_info.get('Version', '')}, {os_info.get('OSArchitecture', '')}.",
              f"- CPU: {cpu.get('Name', 'unknown')}, {cpu.get('NumberOfCores', '?')} cores / {cpu.get('NumberOfLogicalProcessors', '?')} logical processors.",
              f"- RAM: {os_info.get('TotalVisibleMemorySize', 0) / 1024**2:.1f} GiB visible.",
              f"- Graphics: {graphics.get('Name', 'unknown')}; no NVIDIA GPU detected.",
              "- Reused working Python 3.12.10, Git 2.55.0.windows.3 and VS Code 1.144.0; no reinstallation.",
              "- Python 3.14 also exists on the laptop; this project deliberately uses Python 3.12.",
              "- Initial free space: approximately 32 GiB on C: and 448 GiB on D:; project lives on D:.",
              "- Existing Codex extension was already installed. Windows security settings and execution policy were not changed.",
              "", "## Installed requested extensions", ""]
    lines += [f"- `{value}`" for value in installed.values()]
    lines += ["", "Additional Python/Jupyter dependency extensions were installed by VS Code automatically.",
              "", "## Python dependency verification", "", "| Package | Installed version | Import |", "| --- | --- | --- |"]
    for name, result in local.get("packages", {}).items():
        lines.append(f"| {name} | {result.get('version', 'unknown')} | {result.get('import', 'unknown')} |")
    lines += ["", f"Dependency consistency: `{local.get('pip_check', 'pending')}`.",
              "OpenCV uses opencv-python-headless (`cv2`) to avoid duplicate OpenCV builds. SQLite is included in Python.",
              "SQLite FTS5 and TF-IDF retrieval are verified in memory; no external AI service or persistent database was created.",
              "`requirements.txt` pins the actual Windows environment, including transitive packages and CPU PyTorch.",
              "`requirements-colab.txt` pins portable training dependencies and leaves Colab's matched CUDA torch/torchvision in place.",
              "The Colab training dependency installation is prepared but has not been executed remotely.",
              "Initial imports of Pandas/SciPy were blocked by Windows Application Control (CodeIntegrity event 3077). A fresh Python process subsequently passed every import and the full environment check using the same installed versions. No security setting, policy or package downgrade was applied.",
              "Package download timeouts were resolved using resumable official PyPI wheel downloads with SHA-256 validation, followed by installation from the local cache.",
              "", "## Successful tests", "",
              f"- Infrastructure suite: {tests.get('tests_run', 0)} tests, {tests.get('failures', '?')} failures, {tests.get('errors', '?')} errors.",
              "- Tests use temporary synthetic images and mocked epochs; no real model training or dataset access.",
              "- Coverage includes split reproducibility, duplicate leakage, augmentations, unsafe ZIP rejection, disk capacity, CPU inference, top-k counting, random state serialization, phase changes, early stopping and resume.",
              "- Local CPU inference uses an untrained MobileNetV2 and validates tensor shape and finite values; this measures no species performance.",
              f"- Notebook schemas: {local.get('notebook_schema', 'pending')}; local notebook execution: {local_notebook.get('status', 'pending')}. These are separate from the saved Colab GPU result.",
              "", "## Colab evidence and limits", "",
              f"- Authentication: {colab.get('authentication', 'NOT VERIFIED')}.",
              f"- GPU: {colab.get('gpu_name') or 'NOT VERIFIED'}; CUDA available: {cloud.get('cuda_available', 'unknown')}; GPU tensor test: {cloud.get('cuda_tensor_test', 'unknown')}.",
              f"- Remote PyTorch: `{cloud.get('torch', 'unknown')}`; CUDA build: `{cloud.get('cuda_build', 'unknown')}`.",
              f"- Remote Python: `{cloud.get('python', 'unknown')}`; working directory: `{cloud.get('execution_location', 'unknown')}`.",
              "- The initial Colab assignment error about too many servers was resolved through user intervention before the saved GPU test.",
              "- A saved successful cell verifies cloud execution at that time; it does not guarantee a server is still assigned later.",
              "- Programmatic agent control of the currently connected Colab kernel is unavailable in this session. The official extension exposes UI commands; installing it does not grant this assistant an authenticated notebook execution tool.",
              "- No Drive mount, cloud storage account, permanent cloud dataset, remote tunnel or extracted OAuth credential was configured.",
              "", "## Ready for later / manual workflow", "",
              "1. Put your dataset ZIP in `dataset/` when ready. No setup step waits for it.",
              "2. Use the official VS Code Upload to Colab command for `training/colab_source.zip` and, later, your dataset ZIP.",
              "3. Follow `training/train_colab.ipynb`: verify GPU, unpack source, install portable dependencies, safely extract your uploaded ZIP, set the actual class-folder root and inspect detected classes.",
              "4. Enable START_TRAINING only when you choose to begin. No training has run during setup.",
              "5. Download last.pt periodically through Colab Contents to local models/. It includes the best weights, optimizer, early stopping and RNG state. Runtime-only files can disappear after termination.",
              "6. Download final weed_artifacts.zip and run scripts/import_artifacts.py to restore models/ and evaluation/ locally. Existing output files are protected from overwriting.",
              "7. Reconnect/re-authenticate or choose an available GPU if Colab expires or reaches quota. GPU type and availability remain subject to Google allocation.",
              "", "## Evidence files", "",
              "- `evaluation/setup_verification.json`: imports, versions, CPU forward, SQLite retrieval and pip check.",
              "- `evaluation/setup_tests.json` and `evaluation/setup_tests.txt`: synthetic test results.",
              "- `evaluation/setup_colab_status.json` and saved `training/colab_connection_test.ipynb`: remote GPU evidence.",
              "- `evaluation/setup_local_notebook.json`: local kernel smoke execution.",
              "- `evaluation/setup_extensions.txt`: installed extension versions.",
              "- `COLAB_WORKFLOW.md`: exact transfer, interruption and recovery steps.",
              "", "## Official references", "",
              "- [Colab extension guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide)",
              "- [Colab extension commands](https://github.com/googlecolab/colab-vscode/blob/main/package.json)",
              "- [Colab resource limits](https://research.google.com/colaboratory/faq.html#resource-limits)",
              "- [PyTorch local installation](https://pytorch.org/get-started/locally/)", ""]
    (ROOT / "SETUP_STATUS.md").write_text("\n".join(lines), encoding="utf-8")
    print("Updated SETUP_STATUS.md from saved verification evidence.")


if __name__ == "__main__":
    report()
