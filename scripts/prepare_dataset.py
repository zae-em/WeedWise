"""Validate the provided ZIP, preserve originals, group duplicates, and prepare 224px splits."""
import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import cv2
import imagehash
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.archive import extract_dataset


def sha_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def label_name(raw):
    return "Shepherds Purse" if raw.startswith("Shepherd") else raw


class UnionFind:
    def __init__(self, count):
        self.parents = list(range(count))

    def find(self, i):
        while self.parents[i] != i:
            self.parents[i] = self.parents[self.parents[i]]
            i = self.parents[i]
        return i

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parents[max(a, b)] = min(a, b)


def inspect_image(item):
    path, raw_root = item
    relative = path.relative_to(raw_root)
    result = {"source": relative.as_posix(), "class_name": label_name(relative.parts[0]), "byte_sha256": sha_file(path)}
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.load()
            width, height = image.size
            pixels = np.asarray(image)
            result.update(width=width, height=height, format="PNG", phash=str(imagehash.phash(image)),
                          pixel_sha256=hashlib.sha256(f"{width}x{height}:".encode()+pixels.tobytes()).hexdigest(), status="valid")
        return result
    except Exception as error:
        return {**result, "status": "corrupt", "error": str(error)}


def similarity(a, b):
    # A second conservative check avoids grouping different seedlings solely on a similar pHash.
    a, b = a.astype(np.float64), b.astype(np.float64)
    mean_a, mean_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    covariance = ((a-mean_a)*(b-mean_b)).mean()
    return ((2*mean_a*mean_b+6.5025)*(2*covariance+58.5225))/((mean_a**2+mean_b**2+6.5025)*(var_a+var_b+58.5225))


def duplicate_groups(records, raw_root):
    groups = UnionFind(len(records))
    parent_frames = defaultdict(list)
    for index, record in enumerate(records):
        source = Path(record["source"])
        if re.fullmatch(r"\d+(?:-\d+)?", source.stem):
            parent_frames[(record["class_name"], source.stem.split("-")[0])].append(index)
    for family in parent_frames.values():
        for index in family[1:]:
            groups.union(family[0], index)
    exact = defaultdict(list)
    for index, record in enumerate(records):
        exact[record["pixel_sha256"]].append(index)
    exact_duplicate_rows, conflicts = [], []
    for same in exact.values():
        if len(same) > 1:
            labels = {records[i]["class_name"] for i in same}
            for index in same[1:]:
                groups.union(same[0], index)
                exact_duplicate_rows.append({"source": records[index]["source"], "representative": records[same[0]]["source"], "cross_label": len(labels)>1})
            if len(labels) > 1:
                conflicts.extend(same)
    # At Hamming distance <=4, at least one of five disjoint hash bands must agree.
    buckets = defaultdict(list)
    hashes = [int(record["phash"], 16) for record in records]
    near = []
    previews = {}
    def preview(index):
        if index not in previews:
            with Image.open(raw_root/records[index]["source"]) as image:
                previews[index] = np.asarray(ImageOps.exif_transpose(image).convert("L").resize((64,64)))
        return previews[index]
    boundaries = [(0,13),(13,13),(26,13),(39,13),(52,12)]
    for index, value in enumerate(hashes):
        candidates = set()
        for band, (shift, width) in enumerate(boundaries):
            key = (band, (value >> shift) & ((1 << width)-1))
            candidates.update(buckets[key])
            buckets[key].append(index)
        for previous in candidates:
            if records[index]["pixel_sha256"] == records[previous]["pixel_sha256"]:
                continue
            distance = (value ^ hashes[previous]).bit_count()
            if distance <= 4:
                score = similarity(preview(index), preview(previous))
                if score >= 0.97:
                    groups.union(previous,index)
                    near.append({"a":records[previous]["source"],"b":records[index]["source"],"phash_distance":distance,"similarity":float(score),"cross_label":records[index]["class_name"]!=records[previous]["class_name"]})
    return groups, exact, exact_duplicate_rows, near, set(conflicts)


def split_groups(labels, group_ids, seed=42):
    labels, group_ids = np.asarray(labels), np.asarray(group_ids)
    all_indices = np.arange(len(labels))
    class_count = int(labels.max())+1
    def choose(indices, fraction, split_seed):
        desired = np.bincount(labels[indices], minlength=class_count)*fraction
        best, score = None, float("inf")
        splitter = GroupShuffleSplit(n_splits=160, test_size=fraction, random_state=split_seed)
        for left, right in splitter.split(indices, labels[indices], group_ids[indices]):
            counts = np.bincount(labels[indices[right]], minlength=class_count)
            if (counts==0).any() or (np.bincount(labels[indices[left]],minlength=class_count)==0).any():
                continue
            error = float(np.mean(np.abs(counts-desired)/np.maximum(desired,1)))
            if error < score:
                score, best = error, (indices[left], indices[right])
        if best is None:
            raise ValueError("Insufficient independent groups for three class-complete splits; review grouping")
        return best
    train, holdout = choose(all_indices, 0.30, seed)
    val, test = choose(holdout, 0.50, seed+1)
    assignments = {}
    for name, indices in [("train",train),("val",val),("test",test)]:
        for index in indices:
            assignments[int(index)] = name
    return assignments


def write_csv(path, rows):
    rows=list(rows)
    fields=sorted({key for row in rows for key in row})
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(source):
    data=ROOT/"dataset"
    reports=ROOT/"evaluation/dataset"
    reports.mkdir(parents=True,exist_ok=True)
    copied=data/"Dataset.zip"
    if not copied.exists():
        print("Copying ZIP into project; original remains untouched",flush=True)
        shutil.copy2(source,copied)
    if sha_file(source)!=sha_file(copied):
        raise ValueError("Copied ZIP hash differs from original")
    archive_hash=sha_file(copied)
    prior_manifest=data/"processed/manifest.json"
    prior_summary=reports/"summary.json"
    if prior_manifest.is_file() and prior_summary.is_file():
        previous=json.loads(prior_manifest.read_text(encoding="utf-8"))
        if previous.get("source_archive_sha256")==archive_hash and previous.get("parent_frame_groups_applied"):
            if all((data/"processed"/row["processed_path"]).is_file() for row in previous["records"]):
                print("This exact archive is already validated and processed with final source-frame groups. Reusing existing outputs.",flush=True)
                return
    with zipfile.ZipFile(copied) as archive:
        entries=[entry for entry in archive.infolist() if not entry.is_dir()]
        for entry in entries:
            path=PurePosixPath(entry.filename.replace("\\","/"))
            if path.is_absolute() or ".." in path.parts or any(":" in part for part in path.parts):
                raise ValueError("Unsafe ZIP entry")
        checksum_map={}
        print(f"Validating CRC and SHA256 of {len(entries)} ZIP entries",flush=True)
        for index,entry in enumerate(entries):
            with archive.open(entry) as stream:
                checksum_map[entry.filename]=hashlib.file_digest(stream,"sha256").hexdigest()
            if (index+1)%1000==0: print(f"Archive integrity {index+1}/{len(entries)}",flush=True)
        has_nested=any(entry.filename.startswith("nonsegmentedv2/") for entry in entries)
        selected=[entry for entry in entries if entry.filename.startswith("nonsegmentedv2/")] if has_nested else entries
        redundant=[]
        if has_nested:
            for entry in selected:
                twin=entry.filename.removeprefix("nonsegmentedv2/")
                if twin in checksum_map:
                    if checksum_map[twin]!=checksum_map[entry.filename]:
                        raise ValueError("Archive contains differing nested and root copies; manual inspection required")
                    redundant.append(twin)
        raw=data/"raw"
        raw.mkdir(exist_ok=True)
        for entry in selected:
            relative=entry.filename.removeprefix("nonsegmentedv2/") if has_nested else entry.filename
            target=(raw/relative).resolve()
            if not target.is_relative_to(raw.resolve()): raise ValueError("Unsafe extraction")
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and sha_file(target)==checksum_map[entry.filename]: continue
            with archive.open(entry) as incoming,target.open("wb") as outgoing:
                shutil.copyfileobj(incoming,outgoing)
    image_paths=sorted(p for p in raw.rglob("*") if p.is_file() and p.suffix.lower() in {".png",".jpg",".jpeg"})
    print(f"Decoding {len(image_paths)} canonical images",flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        records=list(pool.map(inspect_image,((path,raw) for path in image_paths)))
    corrupt=[record for record in records if record["status"]!="valid"]
    valid=[record for record in records if record["status"]=="valid"]
    print("Checking exact and conservative near-duplicate groups",flush=True)
    groups, exact, duplicate_rows, near, conflicts=duplicate_groups(valid,raw)
    retained=[]
    used=set()
    for index,record in enumerate(valid):
        if index in conflicts: continue
        if record["pixel_sha256"] in used: continue
        used.add(record["pixel_sha256"])
        retained.append({**record,"group_id":f"g{groups.find(index):06d}"})
    classes=sorted({record["class_name"] for record in retained})
    mapping={name:index for index,name in enumerate(classes)}
    assignments=split_groups([mapping[r["class_name"]] for r in retained],[r["group_id"] for r in retained])
    processed=data/"processed"
    if (processed/"manifest.json").exists():
        old=json.loads((processed/"manifest.json").read_text(encoding="utf-8"))
        if old.get("source_archive_sha256")!=archive_hash:
            raise FileExistsError("Processed output belongs to another ZIP; use a separate version directory")
    for index,record in enumerate(retained):
        split=assignments[index]
        relative=Path(split)/record["class_name"]/f"{record['pixel_sha256'][:24]}.jpg"
        destination=processed/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists():
            with Image.open(raw/record["source"]) as image:
                image=ImageOps.exif_transpose(image).convert("RGB")
                # Letterbox preserves the complete seedling; geometry is not stretched.
                ImageOps.pad(image,(224,224),method=Image.Resampling.LANCZOS,color=(127,127,127)).save(destination,"JPEG",quality=95,subsampling=0)
        record.update(split=split,label=mapping[record["class_name"]],processed_path=relative.as_posix(),processed_sha256=sha_file(destination))
    group_splits=defaultdict(set)
    for row in retained: group_splits[row["group_id"]].add(row["split"])
    assert all(len(splits)==1 for splits in group_splits.values())
    frame_contents=defaultdict(set)
    for row in valid:
        path=Path(row["source"])
        if re.fullmatch(r"\d+(?:-\d+)?",path.stem):frame_contents[(row["class_name"],path.stem.split("-")[0])].add(row["pixel_sha256"])
    parent_family_count=sum(len(contents)>1 for contents in frame_contents.values())
    manifest={"schema_version":2,"source_archive_sha256":archive_hash,"classes":classes,"seed":42,"image_size":224,
              "parent_frame_groups_applied":True,"derived_parent_image_families":parent_family_count,
              "records":retained,"group_metadata":"Numeric filename suffixes share a parent-image group; exact pixel hashes and conservative near duplicates also share groups. Original plant-observation/photographer IDs remain unavailable."}
    (processed/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    (processed/"class_to_idx.json").write_text(json.dumps(mapping,indent=2),encoding="utf-8")
    counts={name:{split:sum(row["class_name"]==name and row["split"]==split for row in retained) for split in ["train","val","test"]} for name in classes}
    summary={"source_zip":str(source),"copied_zip":str(copied),"archive_sha256":archive_hash,"zip_entries":len(entries),
             "zip_bytes":copied.stat().st_size,"uncompressed_bytes":sum(entry.file_size for entry in entries),"all_zip_crc_valid":True,
             "redundant_archive_copy_entries":len(redundant),"canonical_images":len(records),"valid_decoded_images":len(valid),"corrupt_images":len(corrupt),
             "exact_duplicate_images_excluded":len(duplicate_rows),"cross_label_exact_conflict_images":len(conflicts),"near_duplicate_pairs":len(near),
             "cross_label_near_pairs":sum(row["cross_label"] for row in near),"processed_images":len(retained),"classes":classes,"class_count":len(classes),
             "split_counts":dict(Counter(row["split"] for row in retained)),"per_class":counts,"independent_groups":len(group_splits),"group_leakage_detected":False,
             "parent_frame_groups_applied":True,"derived_parent_image_families":parent_family_count,
             "original_width_range":[min(r["width"] for r in valid),max(r["width"] for r in valid)],"original_height_range":[min(r["height"] for r in valid),max(r["height"] for r in valid)]}
    (reports/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    write_csv(reports/"image_manifest.csv",retained)
    write_csv(reports/"corrupt_images.csv",corrupt)
    write_csv(reports/"exact_duplicates.csv",duplicate_rows)
    write_csv(reports/"near_duplicates.csv",near)
    write_csv(reports/"label_review.csv",[row for row in near if row["cross_label"]])
    fig,ax=plt.subplots(figsize=(12,6))
    bottom=np.zeros(len(classes))
    for split,color in [("train","#2c7a66"),("val","#e5a445"),("test","#597cbf")]:
        values=np.array([counts[name][split] for name in classes])
        ax.bar(classes,values,bottom=bottom,label=split,color=color)
        bottom+=values
    ax.set_ylabel("Images");ax.set_title("Validated dataset: group-aware split counts")
    ax.tick_params(axis="x",rotation=60);ax.legend();fig.tight_layout();fig.savefig(reports/"class_distribution.png",dpi=150);plt.close(fig)
    contact=Image.new("RGB",(4*224,3*256),"white")
    draw=ImageDraw.Draw(contact)
    for index,name in enumerate(classes):
        row=next(row for row in retained if row["class_name"]==name)
        with Image.open(processed/row["processed_path"]) as image: contact.paste(image,((index%4)*224,(index//4)*256))
        draw.text(((index%4)*224+4,(index//4)*256+228),name,fill="black")
    contact.save(reports/"class_examples.jpg",quality=90)
    package=data/"weed_training_224.zip"
    with zipfile.ZipFile(package,"w",compression=zipfile.ZIP_STORED) as bundle:
        for file in sorted(processed.rglob("*")):
            if file.is_file(): bundle.write(file,file.relative_to(processed).as_posix())
    summary["training_archive_bytes"]=package.stat().st_size
    summary["training_archive_sha256"]=sha_file(package)
    (reports/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    lines=["# Dataset analysis and limitations","",f"Source ZIP SHA256: `{archive_hash}`.","",
           f"Validated {len(entries):,} PNG archive entries with CRC and SHA256. The ZIP stores the same {len(records):,}-image collection twice; {len(redundant):,} redundant entries were excluded after SHA256 comparison.",
           f"Decoded valid images: {len(valid):,}; corrupt: {len(corrupt)}; within-collection exact duplicate rows: {len(duplicate_rows)}; conflicting exact-label images: {len(conflicts)}.",
           f"Retained processed images: {len(retained):,}; classes: {len(classes)}; conservative near-duplicate pairs: {len(near)}.","",
           "| Class | Train | Validation | Test |","| --- | ---: | ---: | ---: |"]
    lines += [f"| {name} | {counts[name]['train']} | {counts[name]['val']} | {counts[name]['test']} |" for name in classes]
    lines += ["","## Reproducibility and integrity","",
              "The original ZIP is preserved outside the project and copied inside dataset/. Raw canonical images are preserved in dataset/raw/. Corrupt images and conflicting exact-label duplicates are excluded from processed training copies and listed for review; originals are retained.",
              "Images are EXIF-corrected, converted to RGB, letterboxed to 224 x 224 and saved as high-quality JPEG. The manifest records all source paths and hashes. Exact pixel copies are represented once; near copies are retained in a shared split group.",
              "Grouping uses shared numeric filename parent IDs (for example 311-1 and 311-2), exact pixel SHA256, and pHash distance <=4 followed by grayscale similarity >=0.97. Splits minimize per-class ratio error over seeded group-shuffle candidates, aiming at 70/15/15. No detected group spans splits. This is not proof that all related observations have been identified.",
              "","## Coverage and generalization limits","",
              "The proposal targets 150-250 weed species plus five crop classes. This dataset has 12 labeled classes: nine weed labels and three crop labels (Common wheat, Maize, Sugar beet). No unsupported labels have been added.",
              "Folder names and the nested nonsegmentedv2 structure match the Aarhus Plant Seedlings Dataset v2. This is provenance evidence, not proof of the ZIP's licensing or acquisition history. Confirm original download terms before redistributing raw images.",
              "The publisher describes approximately 960 plants observed at multiple growth stages. This ZIP has no observation or photographer identifiers; filenames are image numbers and cannot reliably recover plant identities. Residual same-plant/acquisition leakage is therefore possible despite duplicate grouping.",
              "These are cropped seedling photographs with a restricted background domain. Internal held-out accuracy cannot establish field performance, Pakistani crop relevance, adult-plant identification, or independent external validation.",
              "No independent external test set or labeled unknown-species set is available. Calibration and rejection thresholds can be fitted on validation data, but softmax confidence cannot reliably detect every unsupported plant.",
              "Potential incorrect labels are flagged by cross-class duplicate evidence; botanical ground truth requires expert review. No labels were changed based solely on visual similarity.",
              "","Source: [Aarhus Plant Seedlings Dataset](https://vision.eng.au.dk/plant-seedlings-dataset/)",""]
    (reports/"DATASET_REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=Path(r"D:\Haroon FYP\Dataset.zip"))
    main(parser.parse_args().source)
