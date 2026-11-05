"""Queue a harmless calculation; this never allocates a new Colab server."""
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = '''import json, torch
from pathlib import Path
assert Path('/content').is_dir(), 'Select the existing Colab GPU kernel'
assert torch.cuda.is_available(), 'CUDA GPU unavailable; CPU training is disabled'
a=torch.arange(16,device='cuda',dtype=torch.float32).reshape(4,4)
value=(a@a.T).sum().item()
assert value==3680.0
print(json.dumps({'cuda':True,'gpu':torch.cuda.get_device_name(0),'calculation':value}))
'''

if __name__ == '__main__':
    path = ROOT / 'training/expanded_gpu_probe.ipynb'
    path.write_text(json.dumps({'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}},
        'cells': [{'cell_type': 'code', 'id': 'gpu-probe', 'metadata': {},
                   'execution_count': None, 'outputs': [], 'source': SOURCE.splitlines(True)}]}, indent=2))
    request = {'id': 'expanded-gpu-' + str(time.time_ns()), 'action': 'execute',
               'notebook': path.relative_to(ROOT).as_posix(), 'index': 0, 'timeoutSeconds': 90}
    (ROOT / '.weed-control/request.json').write_text(json.dumps(request))
    print(request['id'])
