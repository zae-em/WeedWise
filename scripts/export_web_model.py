"""Export the verified V2 release and attributed seedling examples for the browser."""

import json
import re
import sys
from pathlib import Path

import onnx
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.data import NORMALIZE_MEAN, NORMALIZE_STD
from weed_assistant.inference import PlantClassifier
from weed_assistant.knowledge import KnowledgeBase
from weed_assistant.quality import decode_image


class Probabilities(torch.nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.model = classifier.model
        self.temperature = classifier.temperature

    def forward(self, image):
        return torch.softmax(self.model(image) / self.temperature, dim=1)


def main():
    classifier = PlantClassifier(ROOT / "models/v2/model.pt")
    release = json.loads((ROOT / "models/v2/release.json").read_text(encoding="utf-8"))
    if classifier.model_version != "v2" or classifier.model_hash != release["model_sha256"]:
        raise ValueError("The checkpoint does not match the verified V2 release.")
    kb = KnowledgeBase(ROOT / "knowledge_base/species_v2.json")
    output = ROOT / "web/public/release"
    output.mkdir(parents=True, exist_ok=True)
    size = classifier.artifact["image_size"]
    model_file = output / "model.onnx"
    torch.onnx.export(
        Probabilities(classifier).eval(), (torch.zeros(1, 3, size, size),), str(model_file),
        input_names=["image"], output_names=["probabilities"], opset_version=17,
        dynamo=False, external_data=False,
    )
    onnx.checker.check_model(str(model_file))

    original = ROOT / "dataset/processed"
    manifest = json.loads((original / "manifest.json").read_text(encoding="utf-8"))
    examples = []
    featured = ["Maize", "Fat Hen"]
    labels = featured + [label for label in manifest["classes"] if label not in featured]
    for label in labels:
        row = next(row for row in manifest["records"] if row["split"] == "test" and row["class_name"] == label)
        path = original / row["processed_path"]
        image = path.read_bytes()
        filename = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-") + ".jpg"
        (output / filename).write_bytes(image)
        reference = classifier.predict(decode_image(image))
        examples.append({"label": label, "file": filename, "reference": reference["predictions"]})

    metadata = {
        "version": classifier.model_version, "sha256": classifier.model_hash, "size": size,
        "mean": NORMALIZE_MEAN, "std": NORMALIZE_STD, "threshold": classifier.threshold,
        "classes": [{
            "label": label, "name": kb.get(label)["common_name"], "scientific": kb.get(label)["scientific_name"],
            "family": kb.get(label)["family"], "role": classifier.class_roles[label],
        } for label in classifier.classes],
        "examples": examples,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    print(f"Exported {len(classifier.classes)} classes, {model_file.stat().st_size:,} bytes.")


if __name__ == "__main__":
    main()
