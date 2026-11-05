"""Read saved notebook output without requesting or extracting authentication secrets."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check():
    path = ROOT / "training/colab_connection_test.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    result = {"source": str(path), "authentication": "User reports server connected; not independently verified",
              "gpu_connection": "NOT VERIFIED", "gpu_name": None, "cloud_cell_execution": "NOT VERIFIED",
              "agent_can_execute_connected_kernel": False,
              "reason": "No saved successful GPU test output yet. The current agent has no VS Code kernel execution tool."}
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        chunks = []
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                text = output.get("text", [])
                chunks.append("".join(text) if isinstance(text, list) else text)
        text = "\n".join(chunks)
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                output, _ = json.JSONDecoder().raw_decode(text[index:])
            except ValueError:
                continue
            if not isinstance(output, dict) or "cuda_available" not in output:
                continue
            # Keep only explicitly expected non-sensitive diagnostic fields.
            allowed = ["python", "platform", "torch", "cuda_build", "cuda_available", "gpu_name", "execution_location", "is_colab_runtime", "cuda_tensor_test"]
            result["test_output"] = {key: output[key] for key in allowed if key in output}
            remote = output.get("execution_location", "").startswith("/content") and "Linux" in output.get("platform", "")
            successful = output.get("cuda_available") is True and output.get("cuda_tensor_test") is True
            error = any(item.get("output_type") == "error" for item in cell.get("outputs", []))
            executed = cell.get("execution_count") is not None
            if remote and executed and not error:
                result["authentication"] = "VERIFIED indirectly by successful saved cloud cell execution"
                result["cloud_cell_execution"] = "VERIFIED through VS Code notebook output (user executed)"
                result["gpu_connection"] = "VERIFIED" if successful else "CUDA test failed"
                result["gpu_name"] = output.get("gpu_name")
                result["reason"] = "Saved remote notebook output; connection is a point-in-time check, not a guarantee the runtime remains assigned."
            elif executed:
                result["reason"] = "Saved output exists but does not establish a successful Colab GPU execution."
    target = ROOT / "evaluation/setup_colab_status.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    check()
