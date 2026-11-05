"""Instance crops, duplicate components and acquisition-aware immutable splits."""
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def perceptual_hash(image):
    gray = np.asarray(image.convert('L').resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    values = cv2.dct(gray)[:8, :8].flatten()
    bits = values > np.median(values[1:])
    return int(''.join('1' if b else '0' for b in bits), 2)


class BKTree:
    """Hamming-distance index: do not compare every crop against every crop."""
    def __init__(self):
        self.root = None

    def insert(self, value, index):
        if self.root is None:
            self.root = (value, [index], {})
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(index)
                return
            if distance not in node[2]:
                node[2][distance] = (value, [index], {})
                return
            node = node[2][distance]

    def find(self, value, radius=4):
        stack = [self.root] if self.root else []
        while stack:
            node = stack.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                yield from node[1]
            stack.extend(child for d, child in node[2].items() if distance-radius <= d <= distance+radius)


class UnionFind:
    def __init__(self, count):
        self.parents = list(range(count))

    def find(self, index):
        while self.parents[index] != index:
            self.parents[index] = self.parents[self.parents[index]]
            index = self.parents[index]
        return index

    def join(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parents[max(left, right)] = min(left, right)


def box_pixels(box, width, height):
    if len(box) != 4 or not np.isfinite(box).all():
        raise ValueError('Invalid bounding box')
    x, y, w, h = box
    if min(w, h) <= 0 or min(x, y) < -.001 or x+w > 1.001 or y+h > 1.001:
        raise ValueError('Bounding box outside image')
    return (max(0, round(x*width)), max(0, round(y*height)),
            min(width, round((x+w)*width)), min(height, round((y+h)*height)))


def overlap_fraction(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    return intersection / max(1, min((a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])))


def extract_instance(image, mask, detection, all_detections, taxonomy):
    raw_id = str(detection.get('label_id'))
    taxon = taxonomy.get(raw_id)
    if not taxon or not taxon['included']:
        raise ValueError('unresolved_taxonomy')
    if detection['label'] != taxon['raw_label']:
        raise ValueError('label_id_name_mismatch')
    width, height = image.size
    box = box_pixels(detection['bounding_box'], width, height)
    x1, y1, x2, y2 = box
    if min(x2-x1, y2-y1) < 32:
        raise ValueError('tiny_box')
    for other in all_detections:
        if other is detection:
            continue
        if overlap_fraction(box, box_pixels(other['bounding_box'], width, height)) > .15:
            raise ValueError('overlapping_instance')
    ids = [int(k) for k, v in taxonomy.items() if v.get('included') and v.get('scientific_name') == taxon['scientific_name']]
    region = mask[y1:y2, x1:x2]
    target = np.isin(region, ids)
    vegetation = region != 0
    if target.sum() < 128 or target.mean() < .10:
        raise ValueError('insufficient_target_mask')
    if target.sum() / max(1, vegetation.sum()) < .90:
        raise ValueError('mixed_species_mask')
    # Separate leaves of one plant can be disconnected in a semantic mask.
    # Preserve all meaningful target components inside this isolated bbox;
    # overlapping annotated neighbours were rejected above.
    count, components, stats, _ = cv2.connectedComponentsWithStats(target.astype(np.uint8), 8)
    if count<2:raise ValueError('empty_mask_components')
    selected=np.isin(components,1+np.flatnonzero(stats[1:,cv2.CC_STAT_AREA]>=4))
    if 'stem_x' in detection and not (x1<=detection['stem_x']<=x2 and y1<=detection['stem_y']<=y2):
        raise ValueError('stem_outside_bbox')
    pixels = np.asarray(image)[y1:y2, x1:x2].copy()
    pixels[~selected] = 127
    crop = ImageOps.pad(Image.fromarray(pixels), (224, 224), method=Image.Resampling.LANCZOS, color=(127,127,127))
    return crop, taxon, box, float(target.mean())


def assign_groups(rows, seed=42):
    """Freeze every old split; union sessions and cross-source near duplicates.

    When a component touches old validation/test, new crops inherit that split.
    If it touches incompatible frozen old splits, quarantine the new component.
    Classes without independent train/val/test components are removed from v2.
    """
    uf = UnionFind(len(rows))
    keys = {}
    hashes = BKTree()
    source_hashes = BKTree()
    indexed_sources = set()
    for index, row in enumerate(rows):
        identifiers=[('source', row['source_image']), ('session', row['session_id'])]
        if row.get('source_sha256'):identifiers.append(('source_sha256',row['source_sha256']))
        for key in identifiers:
            if key in keys:
                uf.join(index, keys[key])
            else:
                keys[key] = index
        for other in hashes.find(int(row['phash'], 16), 4):
            uf.join(index, other)
        hashes.insert(int(row['phash'], 16), index)
        if row['source_image'] not in indexed_sources and row.get('source_phash'):
            for other in source_hashes.find(int(row['source_phash'],16),4):uf.join(index,other)
            source_hashes.insert(int(row['source_phash'],16),index)
            indexed_sources.add(row['source_image'])
    components = defaultdict(list)
    for index, row in enumerate(rows):
        components[uf.find(index)].append(row)
    units = list(components.values())
    random.Random(seed).shuffle(units)
    totals = Counter(row['class_name'] for row in rows)
    allocations = {split: Counter() for split in ['train','val','test']}
    fractions = {'train': .70, 'val': .15, 'test': .15}
    rejected = []
    for number, unit in enumerate(units):
        frozen = {r['split'] for r in unit if r['origin'] == 'original'}
        if len(frozen) > 1:
            for row in unit:
                if row['origin'] != 'original':
                    row['excluded'] = 'duplicate_component_touches_multiple_original_splits'
                    rejected.append(row)
            continue
        counts = Counter(r['class_name'] for r in unit)
        # Choose the split with greatest relative per-species deficit.
        split = next(iter(frozen)) if frozen else max(fractions, key=lambda s: sum(
            n * (fractions[s]*totals[c]-allocations[s][c]) / max(1, fractions[s]*totals[c]) for c,n in counts.items()))
        for row in unit:
            row['split'] = split
            row['group_id'] = f'combined-{number:06d}'
        allocations[split].update(counts)
    return rejected


def prepare(root):
    root = Path(root).resolve()
    raw = root / 'dataset/cropandweed/huggingface/hub/Voxel51/CropAndWeed'
    status_path = root / 'dataset/cropandweed/download_status.json'
    status = json.loads(status_path.read_text())
    if (root / '.venv-data/Scripts/python.exe').is_file():
        # The already-running Windows supervisor can reach this function while
        # the downloader is still importing the verified snapshot into FiftyOne.
        if status['state'] == 'verified':
            import psutil
            for _ in range(120):
                if status['state'] != 'verified':
                    break
                if not any('scripts/download_cropandweed.py' in (p.info['cmdline'] or [])
                           for p in psutil.process_iter(['cmdline'], ad_value=None)):
                    break
                time.sleep(5)
                status = json.loads(status_path.read_text())
            status = json.loads(status_path.read_text())
        if status['state'] != 'complete':
            raise RuntimeError('FiftyOne import did not complete; see cropandweed_download.log')
        from scripts.audit_cropandweed_annotations import main as audit_annotations
        audit_annotations()
        audit = json.loads((root / 'evaluation/cropandweed_annotation_audit.json').read_text())
        if audit['issues'] or audit['source_images'] != status.get('fiftyone_samples'):
            raise ValueError('Annotation issues or FiftyOne sample-count mismatch; see the annotation audit')
    if status['state'] not in {'complete', 'verified'}:
        raise RuntimeError('Full dataset download and integrity verification must complete first')
    taxonomy = json.loads((root / 'dataset/cropandweed/taxonomy.json').read_text())['labels']
    old = json.loads((root / 'dataset/processed/manifest.json').read_text())
    kb = json.loads((root / 'knowledge_base/species.json').read_text())['species']
    old_taxa = {r['class_label']: r for r in kb}
    output = root / 'dataset/combined_v2'
    if (output/'manifest.json').exists():
        saved=json.loads((output/'manifest.json').read_text())
        if saved['source_revision']!=status['revision']:
            raise ValueError('Existing combined dataset belongs to a different revision')
        if len(saved['classes'])<=len(saved['original_classes']):
            raise ValueError('Existing combined manifest has no retained new species')
        for row in saved['records']:
            if sha256(output/row['path'])!=row['sha256']:
                raise ValueError('Existing combined dataset integrity failed')
        print('Retaining verified existing combined manifest',saved['fingerprint'],flush=True)
        return saved
    crops = output / 'images'
    crops.mkdir(parents=True, exist_ok=True)
    rows, rejected = [], []
    for row in old['records']:
        if row.get('status') != 'valid' or 'processed_path' not in row:
            continue
        source = root / 'dataset/processed' / row['processed_path']
        if sha256(source) != row['processed_sha256']:
            raise RuntimeError(f'Original image changed: {source}')
        target = crops / ('old-' + row['processed_sha256'] + '.jpg')
        if not target.exists():
            target.write_bytes(source.read_bytes())
        with Image.open(source) as image:
            phash = f'{perceptual_hash(image):016x}'
        taxon = old_taxa[row['class_name']]
        rows.append({'class_name': row['class_name'], 'scientific_name': ' '.join(taxon['scientific_name'].split()[:2]),
                     'original_scientific_name': taxon['scientific_name'],
                     'role': taxon['role'], 'origin': 'original', 'split': row['split'],
                     'path': target.relative_to(output).as_posix(), 'sha256': row['processed_sha256'],
                     'phash': phash, 'source_image': 'original:' + row['source'],
                     'source_phash':phash, 'source_sha256':row['byte_sha256'],
                     'session_id': 'original:' + row['group_id'], 'group_id': 'original:' + row['group_id']})
    original_count = len(rows)
    payload = json.loads((raw / 'samples.json').read_text())
    samples = payload['samples'] if isinstance(payload, dict) else payload
    byte_hashes = {r['sha256']: r for r in rows}
    for number, sample in enumerate(samples):
        name = Path(sample['filepath']).name
        image_path = raw / 'data' / name
        mask_name = Path(sample['segmentation']['mask_path']).name
        mask_path = raw / 'fields/segmentation' / mask_name
        try:
            with Image.open(image_path) as opened:
                image = opened.convert('RGB')
            with Image.open(mask_path) as opened:
                mask = np.asarray(opened)
            if mask.shape != (image.height, image.width):
                raise ValueError('mask_size_mismatch')
        except Exception as error:
            rejected.append({'source_image': name, 'reason': str(error)})
            continue
        source_digest = sha256(image_path)
        source_phash = f'{perceptual_hash(image):016x}'
        detections = (sample.get('ground_truth') or {}).get('detections', [])
        for index, detection in enumerate(detections):
            identity = f'{Path(name).stem}-{index:05d}'
            try:
                crop, taxon, box, coverage = extract_instance(image, mask, detection, detections, taxonomy)
            except ValueError as error:
                rejected.append({'source_image': name, 'instance': index, 'reason': str(error), 'label': detection.get('label')})
                continue
            target = crops / (identity + '.jpg')
            crop.save(target, 'JPEG', quality=95)
            digest = sha256(target)
            row = {'class_name': taxon['class_label'], 'scientific_name': taxon['scientific_name'],
                   'role': taxon['role'], 'origin': 'cropandweed', 'path': target.relative_to(output).as_posix(),
                   'sha256': digest, 'phash': f'{perceptual_hash(crop):016x}',
                   'source_image': 'cropandweed:' + name,
                   'session_id': 'cropandweed:' + name.split('-')[0] + ':' + str(sample['session']),
                   'raw_label': detection['label'], 'raw_label_id': detection['label_id'],
                   'instance_id': identity, 'bbox_pixels': box, 'mask_coverage': coverage,
                   'source_sha256': source_digest, 'source_phash': source_phash}
            if digest in byte_hashes:
                row['excluded'] = 'exact_duplicate'
                previous=byte_hashes[digest]
                if previous['class_name']!=row['class_name']:
                    row['excluded']='conflicting_exact_labels'
                    if previous['origin']!='original':
                        previous['excluded']='conflicting_exact_labels'
                        rejected.append(previous)
                rejected.append(row)
            rows.append(row)
            byte_hashes.setdefault(digest, row)
        if number % 200 == 0:
            print('Parsed sources', number, '/', len(samples), 'retained candidates', len(rows), flush=True)
    rejected.extend(assign_groups(rows))
    active = [r for r in rows if not r.get('excluded')]
    coverage = defaultdict(set)
    for row in active:
        coverage[row['class_name']].add(row['split'])
    excluded_classes = {c: sorted(s) for c,s in coverage.items() if s != {'train','val','test'} and c not in old['classes']}
    for row in active:
        if row['class_name'] in excluded_classes:
            row['excluded'] = 'insufficient_independent_split_coverage'
            rejected.append(row)
    active = [r for r in active if not r.get('excluded')]
    # New output labels append to the unchanged v1 order for compatible head rows.
    classes = old['classes'] + sorted(set(r['class_name'] for r in active)-set(old['classes']))
    for row in active:
        row['label'] = classes.index(row['class_name'])
    assert sum(r['origin']=='original' for r in active) == original_count
    assert sum(r['origin']=='original' and r['split']=='test' for r in active) == 826
    counts = {s: sum(r['split']==s for r in active) for s in ['train','val','test']}
    manifest = {'schema_version': 3, 'classes': classes, 'original_classes': old['classes'],
                'source_revision': status['revision'], 'records': active,
                'split_policy': 'recording_set + session + source image + pHash<=4 components; frozen original splits',
                'preprocessing': 'new: isolated bbox, target semantic components, gray background, letterbox224; original: unchanged',
                'excluded_classes': excluded_classes}
    serialized = json.dumps(manifest, sort_keys=True)
    manifest['fingerprint'] = hashlib.sha256(serialized.encode()).hexdigest()
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    (output/'rejected.json').write_text(json.dumps(rejected, indent=2))
    summary = {'classes': classes, 'class_count': len(classes), 'original_count': len(old['classes']),
               'new_weeds': sorted({r['class_name'] for r in active if r['class_name'] not in old['classes'] and r['role']=='weed'}),
               'new_crops': sorted({r['class_name'] for r in active if r['class_name'] not in old['classes'] and r['role']=='crop'}),
               'overlaps': sorted({r['class_name'] for r in active if r['origin']=='cropandweed' and r['class_name'] in old['classes']}),
               'counts': counts, 'rejections': dict(Counter(r.get('reason',r.get('excluded')) for r in rejected)),
               'per_class': {c: dict(Counter(r['split'] for r in active if r['class_name']==c)) for c in classes},
               'original_test_preserved': True, 'fingerprint': manifest['fingerprint']}
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    (output/'DATASET_REPORT.md').write_text('# Combined species classification dataset\n\n'+
        json.dumps(summary,indent=2)+'\n\nSemantic masks are class masks, not instance masks. Ambiguous overlap and mixed-species masks were excluded. Disconnected leaves inside isolated boxes were preserved.\n'+
        'Old and new image domains differ. Session grouping cannot prove independence across repeated visits to an unidentified farm.\n')
    print(json.dumps(summary, indent=2), flush=True)
    if len(classes)<=len(old['classes']):
        raise RuntimeError('No new species passed independent split and quality checks; v2 training is not justified')
    return manifest


if __name__ == '__main__':
    prepare(Path(__file__).resolve().parents[1])
