"""Import a browser-downloaded v2 bundle with its genuine SHA256 transfer receipt."""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open('rb') as source:return hashlib.file_digest(source,'sha256').hexdigest()


def main(archive,receipt):
    archive,receipt=Path(archive).resolve(),Path(receipt).resolve()
    transfer=json.loads(receipt.read_text())
    if archive.stat().st_size!=transfer['bytes'] or digest(archive)!=transfer['sha256']:
        raise ValueError('Downloaded archive does not match the exported transfer receipt')
    with zipfile.ZipFile(archive) as saved:
        if saved.testzip() is not None:raise ValueError('Artifact archive CRC failed')
        # A separately rebuilt local candidate must not replace the trained manifest.
        combined=ROOT/'dataset/combined_v2'
        if combined.exists():
            current=combined/'manifest.json'
            if not current.is_file():raise RuntimeError('Local combined preparation is incomplete; wait before importing')
            local=json.loads(current.read_text())
            remote=json.loads(saved.read('dataset/combined_v2/manifest.json'))
            if local['fingerprint']!=remote['fingerprint']:
                if (ROOT/'models/v2/model.pt').exists():raise FileExistsError('An existing trained v2 dataset differs; choose a separate release directory')
                backup=ROOT/('dataset/combined_v2_local_candidate_'+local['fingerprint'][:12])
                if not combined.resolve().is_relative_to(ROOT) or not backup.resolve().is_relative_to(ROOT):
                    raise ValueError('Unsafe candidate backup path')
                if backup.exists():raise FileExistsError(backup)
                combined.rename(backup)
        for entry in saved.infolist():
            target=(ROOT/entry.filename).resolve()
            if not target.is_relative_to(ROOT) or not any(entry.filename.startswith(prefix) for prefix in ['models/v2/','evaluation/v2/','dataset/combined_v2/']):
                raise ValueError('Unsafe artifact entry: '+entry.filename)
            if entry.is_dir():continue
            if target.exists():
                if digest(target)==hashlib.sha256(saved.read(entry)).hexdigest():continue
                raise FileExistsError('Refusing to overwrite a differing artifact: '+str(target))
            saved.extract(entry,ROOT)
    destination=ROOT/'models/v2/expanded_artifacts.zip'
    if archive!=destination.resolve():
        if destination.exists() and digest(destination)!=transfer['sha256']:raise FileExistsError(destination)
        if not destination.exists():shutil.copy2(archive,destination)
    (ROOT/'evaluation/v2/import_integrity.json').write_text(json.dumps(transfer,indent=2))
    print('Full archive hash and CRC verified; imported v2 outputs. Run scripts/release_expanded_model.py for full local inference verification.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive');parser.add_argument('--receipt',required=True)
    args=parser.parse_args();main(args.archive,args.receipt)
