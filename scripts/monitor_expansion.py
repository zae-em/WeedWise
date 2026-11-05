"""Read genuine remote progress, retrieve immutable recovery snapshots and final outputs.

Uses supported notebook execution when the official filesystem provider fails.
No Google Drive, exposed network service, credentials, or private Colab API is used.
"""
import base64
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.notebook_transport import execute

REMOTE='/content/weed_expansion'


def read_json(path):
    text=execute(f"from pathlib import Path\nprint('JSON:'+Path({path!r}).read_text())")
    payload=text.split('JSON:',1)[1].rsplit('\ntransport-',1)[0]
    return json.loads(payload)


def download(remote,local):
    """Snapshot a remote artifact, resume chunk transfer, and verify full SHA256."""
    info=read_json_code(f'''import hashlib, json
from pathlib import Path
p=Path({remote!r})
with p.open('rb') as stream:h=hashlib.file_digest(stream,'sha256').hexdigest()
print('JSON:'+json.dumps({{'bytes':p.stat().st_size,'sha256':h}}))''')
    local=Path(local);local.parent.mkdir(parents=True,exist_ok=True)
    if local.exists() and local.stat().st_size==info['bytes']:
        with local.open('rb') as source:
            if hashlib.file_digest(source,'sha256').hexdigest()==info['sha256']:return info
    partial=local.with_suffix(local.suffix+'.downloading')
    offset=partial.stat().st_size if partial.exists() else 0
    if offset>info['bytes']:raise ValueError('Partial file exceeds remote length')
    with partial.open('ab') as target:
        while offset<info['bytes']:
            text=execute(f'''import base64,hashlib
from pathlib import Path
with Path({remote!r}).open('rb') as source:
    source.seek({offset});content=source.read({2*1024**2})
print('HASH:'+hashlib.sha256(content).hexdigest())
print('DATA:'+base64.b64encode(content).decode())''')
            expected=text.split('HASH:',1)[1].splitlines()[0]
            data=base64.b64decode(text.split('DATA:',1)[1].splitlines()[0],validate=True)
            if hashlib.sha256(data).hexdigest()!=expected or not data:raise ValueError('Chunk integrity failed')
            target.write(data);target.flush()
            offset+=len(data)
    with partial.open('rb') as source:
        if hashlib.file_digest(source,'sha256').hexdigest()!=info['sha256']:raise ValueError('Complete file integrity failed')
    partial.replace(local)
    return info


def read_json_code(source):
    text=execute(source)
    return json.loads(text.split('JSON:',1)[1].rsplit('\ntransport-',1)[0])


def retrieve_final():
    transfer=read_json(REMOTE+'/evaluation/expansion_transfer.json')
    folder=ROOT/'models/v2/import_parts';folder.mkdir(parents=True,exist_ok=True)
    for part in transfer['parts']:
        print('Retrieving',part['name'],flush=True)
        info=download(REMOTE+'/'+part['name'],folder/part['name'])
        if info['sha256']!=part['sha256']:raise ValueError('Exported part differs from manifest')
    archive=ROOT/'models/v2/expanded_artifacts.zip'
    with archive.with_suffix('.joining').open('wb') as target:
        for part in transfer['parts']:
            with (folder/part['name']).open('rb') as source:
                while content:=source.read(1024**2):target.write(content)
    joined=archive.with_suffix('.joining')
    with joined.open('rb') as source:
        if hashlib.file_digest(source,'sha256').hexdigest()!=transfer['sha256']:raise ValueError('Final archive SHA256 mismatch')
    joined.replace(archive)
    with zipfile.ZipFile(archive) as saved:
        if saved.testzip() is not None:raise ValueError('Artifact archive CRC failed')
        for entry in saved.infolist():
            path=(ROOT/entry.filename).resolve()
            if not path.is_relative_to(ROOT) or not any(entry.filename.startswith(prefix) for prefix in ['models/v2/','evaluation/v2/','dataset/combined_v2/']):
                raise ValueError('Unsafe artifact archive entry')
            if path.exists():
                if path.is_file() and hashlib.sha256(path.read_bytes()).digest()==hashlib.sha256(saved.read(entry)).digest():continue
                raise FileExistsError(f'Refusing to replace a differing local artifact: {path}')
            saved.extract(entry,ROOT)
    (ROOT/'evaluation/v2/import_integrity.json').write_text(json.dumps(transfer,indent=2))
    print('LOCAL_V2_ARTIFACTS_VERIFIED',flush=True)


def main():
    previous=None
    while True:
        job=read_json(REMOTE+'/evaluation/expansion_job.json')
        (ROOT/'evaluation/expansion_remote_status.json').write_text(json.dumps(job,indent=2))
        if job['state']!=previous:
            print(json.dumps(job),flush=True);previous=job['state']
        if job['state']=='error':raise RuntimeError(job['traceback'])
        if job['state']=='complete':
            retrieve_final();return
        if job['state']=='training':
            progress=read_json(REMOTE+'/models/v2/training_status.json')
            (ROOT/'evaluation/expansion_training_progress.json').write_text(json.dumps(progress,indent=2))
            epochs=progress.get('epochs',0)
            snapshot=epochs//5*5
            if snapshot and not (ROOT/f'models/v2/recovery/epoch_{snapshot:04d}.pt').exists():
                print('Retrieving immutable recovery snapshot for epoch',snapshot,flush=True)
                download(REMOTE+f'/models/v2/recovery/epoch_{snapshot:04d}.pt',ROOT/f'models/v2/recovery/epoch_{snapshot:04d}.pt')
            print(json.dumps(progress),flush=True)
        time.sleep(30)


if __name__=='__main__':main()
