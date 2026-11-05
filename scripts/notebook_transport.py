"""Hashed chunks through public VS Code notebook execution, without credential access.

Fallback for a broken Colab filesystem provider. Only the dedicated expansion probe
notebook is replaced. Each execution must return its unique acknowledgement.
"""
import base64
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.watch_colab_job import request

NOTEBOOK=ROOT/'training/expanded_gpu_probe.ipynb'


def execute(source,timeout=180):
    nonce='transport-'+str(time.time_ns())
    code=source+'\nprint('+repr(nonce)+')\n'
    result=request('replaceExecute',timeout=timeout+15,notebook=NOTEBOOK.relative_to(ROOT).as_posix(),index=0,source=code,timeoutSeconds=timeout)
    texts=[item.get('text','') for output in result.get('outputs',[]) for item in output.get('items',[])]
    text='\n'.join(texts)
    if not result.get('success') or nonce not in text:
        raise RuntimeError('Notebook acknowledgement missing; no transfer assumed: '+text[:2000])
    return text


def upload(path,remote_name=None):
    path=Path(path)
    name=remote_name or path.name
    if Path(name).name!=name:raise ValueError('Remote upload must be a basename')
    chunk_size=2*1024**2
    with path.open('rb') as source:
        index=0
        while content:=source.read(chunk_size):
            digest=hashlib.sha256(content).hexdigest()
            code=f'''import base64, hashlib
from pathlib import Path
folder=Path('/content/weed_transfer');folder.mkdir(exist_ok=True)
target=folder/{(name+'.chunk'+str(index))!r}
content=base64.b64decode({base64.b64encode(content).decode()!r})
assert hashlib.sha256(content).hexdigest()=={digest!r}
target.write_bytes(content)
print('CHUNK_OK',target.name,len(content))
'''
            execute(code)
            index+=1
    with path.open('rb') as source:expected=hashlib.file_digest(source,'sha256').hexdigest()
    code=f'''import hashlib
from pathlib import Path
target=Path('/content')/{name!r}
with target.with_suffix(target.suffix+'.joining').open('wb') as output:
    for i in range({index}):output.write((Path('/content/weed_transfer')/({name!r}+'.chunk'+str(i))).read_bytes())
temporary=target.with_suffix(target.suffix+'.joining')
with temporary.open('rb') as source:assert hashlib.file_digest(source,'sha256').hexdigest()=={expected!r}
temporary.replace(target)
print('FILE_OK',target.name,target.stat().st_size)
'''
    return execute(code)


if __name__=='__main__':
    print(execute("import torch\nfrom pathlib import Path\nassert Path('/content').is_dir() and torch.cuda.is_available()\nprint('TRANSPORT_PROBE_OK',torch.cuda.get_device_name(0))"),flush=True)
