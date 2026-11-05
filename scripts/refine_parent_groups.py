"""Refine existing processed splits using cropped-image filename parent IDs, without re-decoding originals."""
import csv
import json
import re
import shutil
import zipfile
from collections import defaultdict,Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from prepare_dataset import UnionFind,split_groups,sha_file,write_csv

ROOT=Path(__file__).resolve().parents[1]


def main():
    current=(ROOT/"dataset/processed").resolve()
    initial=(ROOT/"dataset/processed_initial").resolve()
    revised=(ROOT/"dataset/processed_revised").resolve()
    reports=ROOT/"evaluation/dataset"
    manifest=json.loads((current/"manifest.json").read_text())
    if manifest.get("parent_frame_groups_applied"):
        cached=json.loads((reports/"summary.json").read_text())
        if cached.get("parent_frame_groups_applied") and cached.get("training_archive_sha256")==sha_file(ROOT/"dataset/weed_training_224.zip"):
            print("Final grouped splits and training archive already verified; reusing them.");return
        finalize(manifest,current,reports)
        print("Parent-frame grouping finalized from existing validated copies.");return
    for path in [current,initial,revised]:
        if not path.is_relative_to((ROOT/"dataset").resolve()):raise ValueError("Output outside dataset workspace")
    if initial.exists() or revised.exists():raise FileExistsError("Split refinement outputs already exist; inspect before proceeding")
    records=manifest["records"]
    groups=UnionFind(len(records))
    known=defaultdict(list)
    by_source={row["source"]:i for i,row in enumerate(records)}
    for i,row in enumerate(records):known[row["group_id"]].append(i)
    for same in known.values():
        for i in same[1:]:groups.union(same[0],i)
    aliases=dict(by_source)
    with (reports/"exact_duplicates.csv").open(newline="",encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["representative"] in by_source:aliases[row["source"]]=by_source[row["representative"]]
    parent=defaultdict(list)
    for source,index in aliases.items():
        path=Path(source)
        if re.fullmatch(r"\d+(?:-\d+)?",path.stem):parent[(records[index]["class_name"],path.stem.split("-")[0])].append(index)
    families=[family for family in parent.values() if len(set(family))>1]
    for family in families:
        for index in family[1:]:groups.union(family[0],index)
    for index,row in enumerate(records):row["group_id"]=f"g{groups.find(index):06d}"
    assignments=split_groups([row["label"] for row in records],[row["group_id"] for row in records],manifest["seed"])
    revised.mkdir()
    for index,row in enumerate(records):
        old_path=current/row["processed_path"]
        split=assignments[index]
        relative=Path(split)/row["class_name"]/old_path.name
        target=revised/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(old_path,target)
        row.update(split=split,processed_path=relative.as_posix())
    manifest.update(schema_version=2,parent_frame_groups_applied=True,derived_parent_image_families=len(families),
                    group_metadata="Shared numeric filename parent IDs, exact pixel hashes and conservative near-duplicate groups. Original longitudinal plant/photographer identifiers unavailable.")
    (revised/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    shutil.copy2(current/"class_to_idx.json",revised/"class_to_idx.json")
    checked=defaultdict(set)
    for row in records:checked[row["group_id"]].add(row["split"])
    assert all(len(values)==1 for values in checked.values())
    current.rename(initial)
    revised.rename(current)
    finalize(manifest,current,reports)


def finalize(manifest,current,reports):
    records=manifest["records"]
    checked=defaultdict(set)
    for row in records:checked[row["group_id"]].add(row["split"])
    assert all(len(values)==1 for values in checked.values())
    families=range(manifest["derived_parent_image_families"])
    summary=json.loads((reports/"summary.json").read_text())
    if not (reports/"initial_split_summary.json").exists():
        (reports/"initial_split_summary.json").write_text(json.dumps(summary,indent=2))
    summary["split_counts"]=dict(Counter(row["split"] for row in records))
    summary["per_class"]={label:{split:sum(row["class_name"]==label and row["split"]==split for row in records) for split in ["train","val","test"]} for label in manifest["classes"]}
    summary.update(independent_groups=len(checked),derived_parent_image_families=len(families),parent_frame_groups_applied=True,group_leakage_detected=False)
    write_csv(reports/"image_manifest.csv",records)
    fig,ax=plt.subplots(figsize=(12,6));bottom=np.zeros(len(manifest["classes"]))
    for split,colour in [("train","#2c7a66"),("val","#e5a445"),("test","#597cbf")]:
        values=np.array([summary["per_class"][label][split] for label in manifest["classes"]]);ax.bar(manifest["classes"],values,bottom=bottom,color=colour,label=split);bottom+=values
    ax.set_ylabel("Images");ax.set_title("Validated groups: exact, near-duplicate and source-frame families");ax.tick_params(axis="x",rotation=60);ax.legend();fig.tight_layout();fig.savefig(reports/"class_distribution.png",dpi=150);plt.close(fig)
    package=ROOT/"dataset/weed_training_224.zip"
    with zipfile.ZipFile(package,"w",compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(current.rglob("*")):
            if path.is_file():archive.write(path,path.relative_to(current).as_posix())
    summary["training_archive_bytes"]=package.stat().st_size;summary["training_archive_sha256"]=sha_file(package)
    (reports/"summary.json").write_text(json.dumps(summary,indent=2))
    report=reports/"DATASET_REPORT.md"
    text=report.read_text(encoding="utf-8")
    table="| Class | Train | Validation | Test |\n| --- | ---: | ---: | ---: |\n"
    for name,counts in summary["per_class"].items():table+=f"| {name} | {counts['train']} | {counts['val']} | {counts['test']} |\n"
    text=re.sub(r"\| Class \| Train.*?(?=\n\n##)",table.strip(),text,flags=re.S)
    text=text.replace("Grouping uses exact pixel SHA256", "Grouping includes shared numeric filename parent IDs (e.g. 311-1 / 311-2), exact pixel SHA256")
    text+="\n## Filename and metadata audit\n\n"
    text+=f"Filename suffixes identified {len(families)} multi-image parent families; these are merged with duplicate groups. The superseded initial split is retained only for audit in dataset/processed_initial/ and must not be used for training.\n\n"
    text+="Final counts: "+json.dumps(summary["split_counts"])+". Final independent detected groups: "+str(len(checked))+".\n"
    text+="\nPNG metadata contained only DPI and GIMP creation comments, not longitudinal observation or photographer identifiers. Shared filename parents are grouped; residual repeated observations across growth stages remain a limitation.\n"
    report.write_text(text,encoding="utf-8")
    print(json.dumps({"split_counts":summary["split_counts"],"groups":len(checked),"parent_families":len(families),"training_archive_sha256":summary["training_archive_sha256"]},indent=2))


if __name__=="__main__":main()
