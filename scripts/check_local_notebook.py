"""Execute a small CPU diagnostic in the project kernel; no training or network access."""
import json
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    source = '''import json, sys
import torch
assert sys.prefix != sys.base_prefix
assert torch.version.cuda is None
value = torch.ones(2, 2) @ torch.ones(2, 2)
assert value.tolist() == [[2.0, 2.0], [2.0, 2.0]]
print(json.dumps({"executable": sys.executable, "virtual_environment": True, "cpu_tensor_test": True}))
'''
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)])
    notebook.metadata.kernelspec = {"name": "ai-weed-assistant", "display_name": "AI Weed Assistant (.venv CPU)", "language": "python"}
    NotebookClient(notebook, kernel_name="ai-weed-assistant", timeout=120,
                   resources={"metadata": {"path": str(ROOT)}}).execute()
    output = "".join(item.get("text", "") for item in notebook.cells[0].outputs if item.get("output_type") == "stream")
    result = json.loads(output)
    assert Path(result["executable"]).resolve() == Path(sys.executable).resolve()
    result["status"] = "PASS"
    nbformat.write(notebook, ROOT / "evaluation/setup_local_notebook.ipynb")
    (ROOT / "evaluation/setup_local_notebook.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
