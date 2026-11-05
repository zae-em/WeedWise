"""Pinned, resumable official full download, integrity verification and FiftyOne import.

Run with .venv-data. No authentication secrets are read or written by this script.
The hub client handles authentication if the user has logged in with `hf auth login`.
"""
import hashlib
import importlib.metadata
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'dataset/cropandweed'
RAW = DEST / 'huggingface/hub/Voxel51/CropAndWeed'
os.environ.setdefault('FIFTYONE_DATABASE_DIR', str(DEST / '.fiftyone/database'))
os.environ.setdefault('FIFTYONE_CONFIG_DIR', str(DEST / '.fiftyone'))
os.environ.setdefault('FIFTYONE_DO_NOT_TRACK', 'true')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('HF_HOME', str(DEST / '.hf'))
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')


def save(name, data):
    DEST.mkdir(parents=True, exist_ok=True)
    target = DEST / name
    temp = target.with_suffix(target.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temp.replace(target)


def main(import_fiftyone=True):
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    api = HfApi()
    info = api.dataset_info('Voxel51/CropAndWeed', files_metadata=True)
    revision_file = DEST / 'revision.json'
    revision = json.loads(revision_file.read_text())['revision'] if revision_file.exists() else info.sha
    info = api.dataset_info('Voxel51/CropAndWeed', revision=revision, files_metadata=True)
    entries = []
    for f in info.siblings:
        lfs = f.lfs
        entries.append({'path': f.rfilename, 'bytes': f.size,
                        'sha256': getattr(lfs, 'sha256', None), 'git_blob': f.blob_id})
    total = sum(e['bytes'] for e in entries)
    free = shutil.disk_usage(DEST).free
    record = {'repo': info.id, 'revision': revision, 'checked_utc': datetime.now(timezone.utc).isoformat(),
              'repository_bytes': total, 'file_count': len(entries), 'disk_free_bytes': free,
              'by_directory_bytes': dict(Counter({prefix: sum(e['bytes'] for e in entries if e['path'].startswith(prefix))
                                                   for prefix in ['data/', 'fields/']})),
              'files': entries, 'license': 'Custom non-commercial research; no redistribution'}
    save('source_manifest.json', record)
    print(json.dumps({k: v for k, v in record.items() if k != 'files'}), flush=True)
    if free < total * 2 + 10 * 1024**3:
        raise OSError('Insufficient free space for raw data, processed crops, and reserve')
    # Official Hub snapshot client resumes incomplete blobs. Pin every file to one commit.
    # Download annotations first so taxonomy and conversion can be inspected while media arrives.
    for name in ['fiftyone.yml', 'metadata.json', 'samples.json', 'README.md']:
        hf_hub_download(info.id, name, repo_type='dataset', revision=revision, local_dir=RAW)
    save('download_status.json', {'state': 'downloading', 'revision': revision, 'expected_bytes': total})
    snapshot_download(info.id, repo_type='dataset', revision=revision, local_dir=RAW, max_workers=2)
    from PIL import Image
    errors = []
    images = masks = 0
    for index, entry in enumerate(entries):
        path = RAW / entry['path']
        if not path.is_file() or path.stat().st_size != entry['bytes']:
            errors.append({'path': entry['path'], 'error': 'missing or size mismatch'})
            continue
        with path.open('rb') as stream:
            sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        if entry['sha256'] and sha != entry['sha256']:
            errors.append({'path': entry['path'], 'error': 'SHA256 mismatch'})
        if path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            try:
                with Image.open(path) as image:
                    image.verify()
                images += entry['path'].startswith('data/')
                masks += entry['path'].startswith('fields/')
            except Exception as error:
                errors.append({'path': entry['path'], 'error': str(error)})
        if index % 500 == 0:
            print('Verified', index, 'of', len(entries), flush=True)
    result = {'state': 'integrity_failed' if errors else 'verified', 'revision': revision,
              'downloaded_bytes': total, 'images': images, 'masks': masks, 'errors': errors,
              'versions': {n: importlib.metadata.version(n) for n in ['huggingface_hub']}}
    save('download_status.json', result)
    if errors:
        raise RuntimeError(f'{len(errors)} integrity errors; see download_status.json')
    if not import_fiftyone:
        print(json.dumps(result, indent=2), flush=True)
        return result
    # Official FiftyOne API, using the already downloaded pinned export and media.
    # local_dir is the documented download destination; avoid a duplicate media cache.
    import fiftyone as fo
    import fiftyone.utils.huggingface as fouh
    fo.config.default_dataset_dir = str(DEST)
    name = 'weed-assistant-cropandweed-' + revision[:8]
    if fo.dataset_exists(name):
        dataset = fo.load_dataset(name)
    elif api.dataset_info(info.id).sha == revision:
        dataset = fouh.load_from_hub(info.id, revision=revision, name=name, persistent=True)
    else:
        # Some FiftyOne versions ignore revision on internal media downloads.
        # Import the fully verified snapshot rather than mixing upstream revisions.
        dataset = fo.Dataset.from_dir(str(RAW), dataset_type=fo.types.FiftyOneDataset, name=name, persistent=True)
    result.update(fiftyone_samples=len(dataset), state='complete')
    result['versions']['fiftyone'] = importlib.metadata.version('fiftyone')
    save('download_status.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        save('download_error.json', {'type': type(error).__name__, 'message': str(error)})
        raise
