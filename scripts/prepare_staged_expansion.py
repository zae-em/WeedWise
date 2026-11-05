"""Prepare a separate dataset from locally verified image/mask pairs."""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.expansion_data import prepare


def main():
    source = ROOT / 'dataset/cropandweed'
    relative_raw = Path('huggingface/hub/Voxel51/CropAndWeed')
    manifest = json.loads((source / 'source_manifest.json').read_text())
    entries = {entry['path']: entry for entry in manifest['files']}
    stage = ROOT / 'dataset/staged_v2' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = stage / 'dataset/cropandweed'
    raw = destination / relative_raw
    raw.mkdir(parents=True)

    def verified(name):
        entry = entries.get(name)
        path = source / relative_raw / name
        if not entry or not entry['sha256'] or not path.is_file():
            return False
        with path.open('rb') as stream:
            return (os.fstat(stream.fileno()).st_size == entry['bytes']
                    and hashlib.file_digest(stream, 'sha256').hexdigest() == entry['sha256'])

    if not verified('samples.json'):
        raise ValueError('The pinned annotation file failed verification')
    payload = json.loads((source / relative_raw / 'samples.json').read_text())
    samples = []
    for sample in payload['samples']:
        image = 'data/' + Path(sample['filepath']).name
        mask = 'fields/segmentation/' + Path(sample['segmentation']['mask_path']).name
        if verified(image) and verified(mask):
            for name in (image, mask):
                target = raw / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    os.link(source / relative_raw / name, target)
            samples.append(sample)
    if not samples:
        raise ValueError('No verified image/mask pairs are available')
    print(f'{len(samples)} verified pairs in {stage}', flush=True)
    (raw / 'samples.json').write_text(json.dumps({**payload, 'samples': samples}), encoding='utf-8')
    (destination / 'download_status.json').write_text(json.dumps({
        'state': 'verified', 'revision': manifest['revision'],
        'scope': 'partial_snapshot', 'verified_pairs': len(samples),
    }), encoding='utf-8')
    (destination / 'taxonomy.json').write_bytes((source / 'taxonomy.json').read_bytes())
    (stage / 'knowledge_base').mkdir()
    os.link(ROOT / 'knowledge_base/species.json', stage / 'knowledge_base/species.json')
    original = ROOT / 'dataset/processed'
    original_manifest = json.loads((original / 'manifest.json').read_text())
    for name in ['manifest.json', *(row['processed_path'] for row in original_manifest['records']
                                     if row.get('status') == 'valid' and 'processed_path' in row)]:
        target = stage / 'dataset/processed' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            os.link(original / name, target)
    prepare(stage)


if __name__ == '__main__':
    main()
