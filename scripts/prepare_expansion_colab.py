"""Create hashed inputs and a Run-All notebook using the existing official provider."""
import gzip
import hashlib
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def digest(path):
    with Path(path).open('rb') as source:return hashlib.file_digest(source,'sha256').hexdigest()


def prepare():
    inputs=ROOT/'training/expansion_inputs'
    inputs.mkdir(exist_ok=True)
    shutil.copy2(ROOT/'models/v1/models/mobilenet_v2/model.pt',inputs/'original_model.pt')
    parts=ROOT/'dataset/colab_parts'
    transfer=json.loads((parts/'transfer_manifest.json').read_text())
    # The current frozen manifest must accompany the archived pixel collection.
    payload={'manifest':json.loads((ROOT/'dataset/processed/manifest.json').read_text()),
             'summary':json.loads((ROOT/'evaluation/dataset/summary.json').read_text())}
    (inputs/'final_splits.json.gz').write_bytes(gzip.compress(json.dumps(payload).encode(),mtime=0))
    source=ROOT/'training/expanded_source.zip'
    with zipfile.ZipFile(source,'w',zipfile.ZIP_DEFLATED) as archive:
        files=list((ROOT/'training').glob('*.py'))+[ROOT/'scripts/__init__.py'] if (ROOT/'scripts/__init__.py').exists() else list((ROOT/'training').glob('*.py'))
        files += [ROOT/'scripts/download_cropandweed.py',ROOT/'scripts/remote_expansion_job.py',
                  ROOT/'knowledge_base/species.json',ROOT/'dataset/cropandweed/revision.json',
                  ROOT/'dataset/cropandweed/metadata.json',ROOT/'dataset/cropandweed/taxonomy.json',
                  ROOT/'dataset/cropandweed/source_manifest.json', inputs/'original_model.pt',inputs/'final_splits.json.gz']
        for file in files:archive.write(file,file.relative_to(ROOT).as_posix())
    upload=['training/expanded_source.zip','dataset/colab_parts/transfer_manifest.json']+['dataset/colab_parts/'+r['name'] for r in transfer['parts']]
    expected={Path(name).name:digest(ROOT/name) for name in upload}
    code=f'''import hashlib, json, os, subprocess, sys, zipfile
from pathlib import Path
import torch
assert torch.cuda.is_available(), 'Select a Colab CUDA GPU; CPU fallback is disabled'
assert Path('/content').is_dir(), 'This notebook requires the temporary Colab runtime'
a=torch.arange(16,device='cuda',dtype=torch.float32).reshape(4,4)
assert (a@a.T).sum().item()==3680.0
print('Verified CUDA:',torch.cuda.get_device_name(0))
expected={expected!r}
for name,expected_hash in expected.items():
    path=Path('/content')/name
    assert path.is_file(), f'Upload prepared input {{name}} with the official Colab extension'
    with path.open('rb') as stream: assert hashlib.file_digest(stream,'sha256').hexdigest()==expected_hash, f'Input hash mismatch: {{name}}'
project=Path('/content/weed_expansion');project.mkdir(exist_ok=True)
with zipfile.ZipFile('/content/expanded_source.zip') as archive:
    for entry in archive.infolist(): assert (project/entry.filename).resolve().is_relative_to(project.resolve())
    archive.extractall(project)
os.chdir(project)
if str(project) not in sys.path:sys.path.insert(0,str(project))
dataset_zip=project/'original_training.zip'
if not dataset_zip.is_file():
    transfer=json.loads(Path('/content/transfer_manifest.json').read_text())
    temporary=dataset_zip.with_suffix('.joining')
    with temporary.open('wb') as target:
        for piece in transfer['parts']:
            target.write((Path('/content')/piece['name']).read_bytes())
    with temporary.open('rb') as stream: assert hashlib.file_digest(stream,'sha256').hexdigest()==transfer['sha256']
    temporary.replace(dataset_zip)
from training.prepare_runtime import prepare_runtime_dataset
prepare_runtime_dataset(dataset_zip,project/'dataset/processed',project/'training/expansion_inputs/final_splits.json.gz')
pid_file=project/'expansion_pid.json'
existing=False
if pid_file.exists():
    pid=json.loads(pid_file.read_text())['pid']
    try: existing=b'scripts.remote_expansion_job' in Path(f'/proc/{{pid}}/cmdline').read_bytes()
    except FileNotFoundError: pass
if existing:
    print('Existing expansion worker retained:',pid)
else:
    log=(project/'expansion.log').open('a',buffering=1)
    process=subprocess.Popen([sys.executable,'-u','-m','scripts.remote_expansion_job'],cwd=project,stdout=log,stderr=subprocess.STDOUT)
    pid_file.write_text(json.dumps({{'pid':process.pid}}))
    print('Expansion worker started:',process.pid,'Monitor evaluation/expansion_job.json and models/v2/training_status.json')
'''
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'accelerator':'GPU'},
        'cells':[{'cell_type':'markdown','id':'instructions','metadata':{},'source':['# Expanded MobileNetV2 training\n','Select the existing T4 kernel, upload the files in expanded_upload_manifest.json using the official extension, then Run All. No Drive is used. The worker downloads the complete pinned dataset, verifies it, isolates instances, groups sessions, preserves the original test set, trains and evaluates. Monitor and retrieve checkpoints using scripts/monitor_expansion.py. Runtime files are temporary until retrieved.\n']},
                 {'cell_type':'code','id':'bootstrap','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(True)}]}
    (ROOT/'training/expanded_training.ipynb').write_text(json.dumps(notebook,indent=2))
    (ROOT/'training/expanded_upload_manifest.json').write_text(json.dumps({'files':upload,'sha256_by_name':expected},indent=2))
    return upload,code


if __name__=='__main__':
    upload,code=prepare()
    print('Prepared',len(upload),'inputs,',sum((ROOT/f).stat().st_size for f in upload),'bytes')
