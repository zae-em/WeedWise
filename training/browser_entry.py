import hashlib, zipfile
from pathlib import Path
import torch
assert torch.cuda.is_available(), 'Select Runtime > Change runtime type > T4 GPU, then reconnect. CPU training is disabled.'
a=torch.arange(16,device='cuda',dtype=torch.float32).reshape(4,4)
assert (a@a.T).sum().item()==3680.0
print('Verified GPU:',torch.cuda.get_device_name(0))
from google.colab import files
inputs=Path('/content/expanded_colab_inputs.zip')
if not inputs.is_file():
    print('Choose the local training/expanded_colab_inputs.zip file. Upload goes to temporary runtime storage only.')
    files.upload()
assert inputs.is_file(), 'Select expanded_colab_inputs.zip'
with inputs.open('rb') as source:assert hashlib.file_digest(source,'sha256').hexdigest()=='aad7c06301f62a4f99e4401416a19a0c610b32aa20cb8d9bcca25e0a00d74ef2', 'Input archive SHA256 mismatch'
with zipfile.ZipFile(inputs) as archive:
    assert archive.testzip() is None
    for entry in archive.infolist():assert (Path('/content')/entry.filename).resolve().is_relative_to(Path('/content'))
    archive.extractall('/content')
print('Input upload verified')


import hashlib, json, os, subprocess, sys, zipfile
from pathlib import Path
import torch
assert torch.cuda.is_available(), 'Select a Colab CUDA GPU; CPU fallback is disabled'
assert Path('/content').is_dir(), 'This notebook requires the temporary Colab runtime'
a=torch.arange(16,device='cuda',dtype=torch.float32).reshape(4,4)
assert (a@a.T).sum().item()==3680.0
print('Verified CUDA:',torch.cuda.get_device_name(0))
expected={'expanded_source.zip': '3ca7e09de88f19b1db463d58e14a93a68d5c424ee2fcf0ecf2c077029416abee', 'transfer_manifest.json': '77deb8ed8e9084347a7d9c4f011a52d49a83c49554b2e92bcf9d9852c32eafe1', 'weed_training_224.zip.part000': '6d3552ef013200d976080787ef506773153bd500895e18312892c9adcf089736', 'weed_training_224.zip.part001': '33b580dd4f2605d5613e1a6f8268acbb59dd705fa395c3e7321ed479cb1049a9', 'weed_training_224.zip.part002': '6f2e4d9b1e3c0742dd5e5cfc795847d45e87b0dd25df850e2336d6005c59073d', 'weed_training_224.zip.part003': '53111ba6d28c20f329fc9734dc6432fb6af66c3ef53a515736c522bae47137ba', 'weed_training_224.zip.part004': '043d72d2478afb69f2c8720269001c57dc3c6c556fbe46e924d9aced643fbae1', 'weed_training_224.zip.part005': 'de0eb0b7b93d221699a6c29227e3d328a473e0a4234b76efcdb5b1e2e43f9bd5', 'weed_training_224.zip.part006': '5f4de8c1f5d5c1bc1c643e07b994e0208b889715384ff7f49e60dea991985f5b', 'weed_training_224.zip.part007': '35e99c914ba29c42f6a9844356c72374d9ca8d914f02cfc2ba6bb9296cd5b2b2', 'weed_training_224.zip.part008': 'fce82be7587397ab046d015e37493c2c12f112a91ce70e6c3f50939356e1775a', 'weed_training_224.zip.part009': 'd6137f54d756e6da38d11343cdcf5f05d9b18d9263624d814702568e325f26ea', 'weed_training_224.zip.part010': '9825831482c3ad6ae08ff217d82cb7190e00683a674f3ad275f113e1fca200f8', 'weed_training_224.zip.part011': 'a898728740f625ef7a3671f57ece19f5c3b2776cd52bd298c9ba7b3616755349', 'weed_training_224.zip.part012': 'd1ddd966c1f9fbfd6491e49f103c101fcd79721b32cd4bf1f5234a8631dacdc3', 'weed_training_224.zip.part013': '02593b8fb9e556d5664e45052f74f5a8d2c633c792af9dc58eb5075490c9f8ab', 'weed_training_224.zip.part014': 'fe285036071e6032ef29a762ee5ab644e258a5f403793a23273316c2f56b717a', 'weed_training_224.zip.part015': '42c00d8314b7ea9478e61edfda88c284fa0f65e8fe217fe5b2d6dfc8d7b5c1ff'}
for name,expected_hash in expected.items():
    path=Path('/content')/name
    assert path.is_file(), f'Upload prepared input {name} with the official Colab extension'
    with path.open('rb') as stream: assert hashlib.file_digest(stream,'sha256').hexdigest()==expected_hash, f'Input hash mismatch: {name}'
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
    try: existing=b'scripts.remote_expansion_job' in Path(f'/proc/{pid}/cmdline').read_bytes()
    except FileNotFoundError: pass
if existing:
    print('Existing expansion worker retained:',pid)
else:
    log=(project/'expansion.log').open('a',buffering=1)
    process=subprocess.Popen([sys.executable,'-u','-m','scripts.remote_expansion_job'],cwd=project,stdout=log,stderr=subprocess.STDOUT)
    pid_file.write_text(json.dumps({'pid':process.pid}))
    print('Expansion worker started:',process.pid,'Monitor evaluation/expansion_job.json and models/v2/training_status.json')


import hashlib, json, time
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
