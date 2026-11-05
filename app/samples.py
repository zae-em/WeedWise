"""Local held-out images for the active model's example workflows."""

import json
from pathlib import Path

import streamlit as st

from weed_assistant.config import ROOT


@st.cache_data(show_spinner=False)
def _index_samples(manifest_path, manifest_modified, path_field):
    dataset = Path(manifest_path).parent.resolve()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    samples = {label: [] for label in manifest["classes"]}
    for row in manifest["records"]:
        if row["split"] != "test":
            continue
        examples = samples[row["class_name"]]
        if len(examples) >= 5:
            continue
        photo = (dataset / row[path_field]).resolve()
        if photo.is_relative_to(dataset) and photo.is_file():
            examples.append((photo, row.get("origin", "original")))
    return samples


def heldout_samples(classifier):
    expanded = classifier.model_version == "v2"
    dataset = ROOT / "dataset" / ("combined_v2" if expanded else "processed")
    manifest = dataset / "manifest.json"
    if not manifest.is_file():
        return {}
    samples = _index_samples(str(manifest), manifest.stat().st_mtime_ns, "path" if expanded else "processed_path")
    return {label: samples[label] for label in classifier.classes if samples.get(label)}
