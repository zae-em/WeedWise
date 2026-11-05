"""Prepare exact local splits on Colab, reusing uploaded pixels when only group assignments changed."""
import gzip
import hashlib
import json
import os
import shutil
from pathlib import Path
from .archive import extract_dataset


def prepare_runtime_dataset(archive,destination,final_metadata=None):
    destination=Path(destination)
    metadata=None
    if final_metadata:
        metadata=json.loads(gzip.decompress(Path(final_metadata).read_bytes()).decode("utf-8"))
    if (destination/"manifest.json").is_file():
        existing=json.loads((destination/"manifest.json").read_text())
        if existing.get("parent_frame_groups_applied"):
            return existing
        raise ValueError("Existing runtime dataset has superseded split assignments")
    if metadata is None:
        extract_dataset(archive,destination)
        manifest=json.loads((destination/"manifest.json").read_text())
        if not manifest.get("parent_frame_groups_applied"):raise ValueError("Final group-aware manifest required")
    else:
        initial=destination.with_name(destination.name+"_initial")
        if not (initial/"manifest.json").is_file():extract_dataset(archive,initial)
        old=json.loads((initial/"manifest.json").read_text())
        manifest=metadata["manifest"]
        indexed={row["pixel_sha256"]:row for row in old["records"]}
        destination.mkdir(exist_ok=True)
        for row in manifest["records"]:
            prior=indexed[row["pixel_sha256"]]
            source=initial/prior["processed_path"]
            target=(destination/row["processed_path"]).resolve()
            if not target.is_relative_to(destination.resolve()):raise ValueError("Unsafe final manifest path")
            with source.open("rb") as stream:
                if hashlib.file_digest(stream,"sha256").hexdigest()!=row["processed_sha256"]:raise ValueError("Uploaded image differs from final local manifest")
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                try:os.link(source,target)
                except OSError:shutil.copy2(source,target)
        (destination/"manifest.json").write_text(json.dumps(manifest,indent=2))
        (destination/"class_to_idx.json").write_text(json.dumps({name:index for index,name in enumerate(manifest["classes"])},indent=2))
    seen={}
    for row in manifest["records"]:
        previous=seen.setdefault(row["group_id"],row["split"])
        if previous!=row["split"]:raise ValueError("A source/duplicate group crosses runtime splits")
    return manifest
