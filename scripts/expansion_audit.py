"""Preserve v1 and probe only the selected notebook through the existing VS Code bridge."""
import hashlib
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    backup = ROOT / 'models/v1'
    backup.mkdir(exist_ok=True)
    inventory = []
    for folder in ['models/mobilenet_v2', 'evaluation/mobilenet_v2']:
        for source in (ROOT / folder).rglob('*'):
            if not source.is_file():
                continue
            destination = backup / folder / source.relative_to(ROOT / folder)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if digest(source) != digest(destination):
                    raise RuntimeError(f'Backup differs: {destination}')
            else:
                shutil.copy2(source, destination)
            inventory.append({'source': source.relative_to(ROOT).as_posix(),
                              'backup': destination.relative_to(ROOT).as_posix(),
                              'bytes': source.stat().st_size, 'sha256': digest(destination)})
    for name in ['dataset/processed/manifest.json', 'knowledge_base/species.json']:
        source = ROOT / name
        target = backup / source.name
        if not target.exists():
            shutil.copy2(source, target)
        if digest(source) != digest(target):
            raise RuntimeError(f'Backup differs: {name}')
    (backup / 'backup_manifest.json').write_text(json.dumps(inventory, indent=2))
    (ROOT / 'models/v2').mkdir(exist_ok=True)
    evidence = json.loads((ROOT / 'evaluation/local_model_verification.json').read_text())
    print(json.dumps({'backup_files': len(inventory), 'v1_model_sha256': digest(ROOT / 'models/mobilenet_v2/model.pt'),
                      'original_evidence': evidence}, indent=2))
    control = ROOT / '.weed-control'
    previous = json.loads((control / 'request.json').read_text())
    result = control / (previous['id'] + '.json')
    if not result.exists() or json.loads(result.read_text()).get('status') == 'running':
        print('Controller has a pending action; no request replaced.')
        return
    request = {'id': 'expansion-status-' + str(time.time_ns()), 'action': 'status'}
    (control / 'request.json').write_text(json.dumps(request))
    print('Queued read-only controller status:', request['id'])


if __name__ == '__main__':
    main()
