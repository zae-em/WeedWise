"""Temporary Colab worker. All permanent outputs are exported for local retrieval."""
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / 'evaluation/expansion_job.json'


def status(state, **kwargs):
    STATUS.parent.mkdir(parents=True,exist_ok=True)
    target=STATUS.with_suffix('.pending')
    target.write_text(json.dumps({'state':state,'updated_unix':time.time(),**kwargs},indent=2))
    target.replace(STATUS)


def main():
    import torch
    if not torch.cuda.is_available() or not Path('/content').is_dir():
        raise RuntimeError('Colab CUDA runtime required')
    torch.set_num_threads(2)
    start=time.time()
    verified_combined = '--verified-combined' in sys.argv[1:]
    gpu = torch.cuda.get_device_name(0)
    status('initializing' if verified_combined else 'downloading', gpu=gpu)
    dependencies = ['scikit-learn', 'scipy', 'matplotlib']
    if not verified_combined:
        dependencies += ['huggingface_hub>=0.36', 'opencv-python-headless']
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', *dependencies], check=True)
    if not verified_combined:
        from scripts.download_cropandweed import main as download
        download(import_fiftyone=False)
        status('preparing_instances', gpu=gpu)
        from training.expansion_data import prepare
        prepare(ROOT)
    from training.expanded_train import train, ExpansionConfig
    config=ExpansionConfig(initial_checkpoint=str(ROOT/'training/expansion_inputs/original_model.pt'),
        data_dir=str(ROOT/'dataset/combined_v2'),output_dir=str(ROOT/'models/v2'),evaluation_dir=str(ROOT/'evaluation/v2'))
    if (ROOT/'models/v2/last.pt').is_file():
        config.resume=str(ROOT/'models/v2/last.pt')
    status('training',gpu=torch.cuda.get_device_name(0))
    metrics=train(config)
    packages = ['torch', 'torchvision', 'numpy', 'Pillow']
    if not verified_combined:
        packages.append('huggingface_hub')
    provenance={'gpu':gpu,'duration_seconds':time.time()-start,
        'versions':{n:importlib.metadata.version(n) for n in packages},
        'source_revision':json.loads((ROOT/'dataset/cropandweed/revision.json').read_text())['revision']}
    (ROOT/'evaluation/v2/provenance.json').write_text(json.dumps(provenance,indent=2))
    status('exporting',**provenance,metrics=metrics)
    archive=ROOT/'expanded_artifacts.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as output:
        folders = ['models/v2', 'evaluation/v2']
        if verified_combined:
            output.write(ROOT/'dataset/combined_v2/manifest.json', 'dataset/combined_v2/manifest.json')
        else:
            folders.append('dataset/combined_v2')
        for folder in folders:
            for file in sorted((ROOT/folder).rglob('*')):
                if file.is_file():
                    output.write(file,file.relative_to(ROOT).as_posix())
    pieces=[]
    with archive.open('rb') as source:
        index=0
        while content:=source.read(8*1024**2):
            name=f'expanded_artifacts.zip.part{index:04d}'
            (ROOT/name).write_bytes(content)
            pieces.append({'name':name,'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()})
            index+=1
    with archive.open('rb') as source:
        digest=hashlib.file_digest(source,'sha256').hexdigest()
    transfer={'name':archive.name,'sha256':digest,'bytes':archive.stat().st_size,'parts':pieces}
    (ROOT/'evaluation/expansion_transfer.json').write_text(json.dumps(transfer,indent=2))
    status('complete',**provenance,metrics=metrics,transfer_manifest='evaluation/expansion_transfer.json')


if __name__=='__main__':
    try:
        os.chdir(ROOT)
        main()
    except BaseException as error:
        status('error',error=str(error),traceback=traceback.format_exc())
        raise
