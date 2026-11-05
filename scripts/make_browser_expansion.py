"""Single-upload, Run-All Colab browser fallback. No Drive mount or copy is used."""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_expansion_colab import prepare


def verified_inputs(inputs):
    combined = ROOT/'dataset/combined_v2'
    manifest = json.loads((combined/'manifest.json').read_text())
    verified = json.loads((ROOT/'evaluation/expansion_local_status.json').read_text())
    if manifest['fingerprint'] != verified['fingerprint']:
        raise ValueError('The combined manifest differs from the verified local preparation')
    source_files = sorted((ROOT/'training').glob('*.py'))
    source_files += [ROOT/'scripts/remote_expansion_job.py', ROOT/'dataset/cropandweed/revision.json', combined/'manifest.json']
    temporary = inputs.with_suffix('.pending')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for file in source_files:
            archive.write(file, file.relative_to(ROOT).as_posix())
        archive.write(ROOT/'models/v1/models/mobilenet_v2/model.pt', 'training/expansion_inputs/original_model.pt')
        for name in sorted({record['path'] for record in manifest['records']}):
            file = (combined/name).resolve()
            if not file.is_relative_to(combined.resolve()):
                raise ValueError(f'Unsafe combined dataset path: {name}')
            archive.write(file, file.relative_to(ROOT).as_posix(), compress_type=zipfile.ZIP_STORED)
    temporary.replace(inputs)
    return '''import json, subprocess, sys
from pathlib import Path
project = Path('/content/weed_expansion')
pid_file = project/'expansion_pid.json'
if pid_file.exists():
    pid = json.loads(pid_file.read_text())['pid']
    command = Path(f'/proc/{pid}/cmdline')
    if command.exists() and b'scripts.remote_expansion_job' in command.read_bytes():
        raise RuntimeError(f'Existing worker {pid} is running. Use the monitor cell to follow it.')
status_file = project/'evaluation/expansion_job.json'
if status_file.exists() and json.loads(status_file.read_text())['state'] == 'complete':
    raise RuntimeError('This runtime already has a completed export. Use the monitor cell to download it.')
with (project/'expansion.log').open('a') as log:
    worker = subprocess.Popen([sys.executable, '-u', '-m', 'scripts.remote_expansion_job', '--verified-combined'],
                              cwd=project, stdout=log, stderr=subprocess.STDOUT)
pid_file.write_text(json.dumps({'pid': worker.pid}))
print('Worker launched. The epoch log will confirm training progress.')
'''


def main(verified=False, output_dir=ROOT/'training'):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if verified:
        inputs=output_dir/'verified_v2_colab_inputs.zip'
        bootstrap=verified_inputs(inputs)
        destination='/content/weed_expansion'
    else:
        files,bootstrap=prepare()
        inputs=output_dir/'expanded_colab_inputs.zip'
        with zipfile.ZipFile(inputs,'w',zipfile.ZIP_STORED) as archive:
            for name in files:archive.write(ROOT/name,Path(name).name)
        destination='/content'
    with inputs.open('rb') as source:digest=hashlib.file_digest(source,'sha256').hexdigest()
    setup=f'''import hashlib, zipfile
from pathlib import Path
import torch
assert torch.cuda.is_available(), 'Select Runtime > Change runtime type > T4 GPU, then reconnect. CPU training is disabled.'
a=torch.arange(16,device='cuda',dtype=torch.float32).reshape(4,4)
assert (a@a.T).sum().item()==3680.0
print('Verified GPU:',torch.cuda.get_device_name(0))
from google.colab import files
inputs=Path('/content')/{inputs.name!r}
if not inputs.is_file():
    print('Choose {inputs.name} from the same folder as this notebook. Upload goes to temporary runtime storage only.')
    files.upload()
assert inputs.is_file(), 'Select {inputs.name}'
with inputs.open('rb') as source:assert hashlib.file_digest(source,'sha256').hexdigest()=={digest!r}, 'Input archive SHA256 mismatch'
destination=Path({destination!r})
with zipfile.ZipFile(inputs) as archive:
    assert archive.testzip() is None
    for entry in archive.infolist():assert (destination/entry.filename).resolve().is_relative_to(destination)
    archive.extractall(destination)
print('Input upload verified')
'''
    monitor='''import hashlib, json, time
from pathlib import Path
from google.colab import files
root=Path('/content/weed_expansion')
sent=set();previous=None
while True:
    status_path=root/'evaluation/expansion_job.json'
    if not status_path.exists():time.sleep(2);continue
    try:status=json.loads(status_path.read_text())
    except json.JSONDecodeError:time.sleep(2);continue
    progress_path=root/'models/v2/training_status.json'
    if progress_path.exists():
        try:progress=json.loads(progress_path.read_text())
        except json.JSONDecodeError:progress={}
    else:progress={}
    marker=(status['state'],progress.get('epochs'))
    if marker!=previous:
        print(json.dumps({'job':status,'training':progress}),flush=True);previous=marker
    # Completed immutable snapshots survive concurrent training writes. These are
    # protected against runtime loss only after the browser has downloaded them.
    for checkpoint in sorted((root/'models/v2/recovery').glob('epoch_*.pt')):
        if checkpoint.name not in sent:
            with checkpoint.open('rb') as source:digest=hashlib.file_digest(source,'sha256').hexdigest()
            print('Recovery snapshot ready:',checkpoint.name,'SHA256',digest,flush=True)
            files.download(str(checkpoint));sent.add(checkpoint.name)
    if status['state']=='error':raise RuntimeError(status['traceback'])
    if status['state']=='complete':
        archive=root/'expanded_artifacts.zip'
        with archive.open('rb') as source:digest=hashlib.file_digest(source,'sha256').hexdigest()
        print('Training and evaluation completed. Export SHA256:',digest,flush=True)
        print('Save the archive and transfer receipt inside your local project. Runtime storage remains temporary.')
        files.download(str(root/'evaluation/expansion_transfer.json'))
        files.download(str(archive))
        break
    time.sleep(30)
'''
    def cell(identity,source):return {'cell_type':'code','id':identity,'metadata':{},'execution_count':None,'outputs':[],'source':source.splitlines(True)}
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'accelerator':'GPU'},
        'cells':[{'cell_type':'markdown','id':'instructions','metadata':{},'source':['# Expanded weed species training — local files, temporary Colab GPU\n',
        f'Open this local notebook through Colab File > Upload notebook. Select the existing T4 runtime and Run All. The first cell asks for `{inputs.name}` from the same local folder as this notebook. All data stays in temporary `/content`; no Google Drive is mounted or used by this workflow.\n',
        'New runs train for 20 to 25 epochs. A resumed run uses its checkpoint schedule.\n',
        'Each epoch uses every designated training image, with bounded minority replay. Validation alone controls early stopping and calibration. The untouched test set is evaluated after training. Checkpoints are offered for download every five completed epochs; keep the browser open and allow the requested downloads. A checkpoint is safe from runtime deletion only after it is downloaded.\n']},cell('verify-and-upload',setup),cell('bootstrap-worker',bootstrap),cell('monitor-and-download',monitor)]}
    notebook_name = 'verified_v2_training_browser.ipynb' if verified else 'expanded_training_browser.ipynb'
    (output_dir/notebook_name).write_text(json.dumps(notebook,indent=2))
    print('Prepared browser notebook and one input archive:',inputs.stat().st_size,'bytes, SHA256',digest)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verified-combined', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'training')
    args = parser.parse_args()
    main(args.verified_combined, args.output_dir)
