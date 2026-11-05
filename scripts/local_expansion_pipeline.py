"""Finish local data preparation after the ongoing resumable download completes.

This supervisor never starts CPU training or overwrites a trained combined manifest.
It may safely run while the independent Colab workflow is awaiting authentication.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import psutil
    status_path=ROOT/'dataset/cropandweed/download_status.json'
    downloader=ROOT/'.venv-data/Scripts/python.exe'
    attempts=0
    retry_at=0
    while True:
        status=json.loads(status_path.read_text()) if status_path.exists() else {}
        if status.get('state') == 'complete':break
        running=False
        for process in psutil.process_iter(['name','cmdline']):
            try:
                command=process.info['cmdline'] or []
                if (process.info['name'] or '').lower() in {'python.exe','python','python3'} and any(Path(arg).name=='download_cropandweed.py' for arg in command):running=True
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        if not running and time.monotonic() >= retry_at:
            if attempts>=4:raise RuntimeError('Download repeatedly failed; see cropandweed_download.log and download_error.json')
            attempts+=1
            retry_at=time.monotonic()+60*2**attempts
            print('Resuming official full download, attempt',attempts,flush=True)
            with (ROOT/'evaluation/cropandweed_download.log').open('a') as output:
                subprocess.Popen([str(downloader),'-u','scripts/download_cropandweed.py'],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
        time.sleep(30)
    if not (ROOT/'dataset/cropandweed/taxonomy.json').is_file():
        subprocess.run([sys.executable,'scripts/resolve_cropandweed_taxonomy.py'],cwd=ROOT,check=True)
    from training.expansion_data import prepare
    manifest=prepare(ROOT)
    progress={'state':'local_dataset_prepared','classes':manifest['classes'],'fingerprint':manifest['fingerprint'],
              'training_started':False,'next':'Reconnect the existing Colab GPU and run scripts/launch_expansion.py; no CPU fallback'}
    destination=ROOT/'evaluation/expansion_local_status.json'
    temporary=destination.with_suffix('.pending')
    temporary.write_text(json.dumps(progress,indent=2))
    temporary.replace(destination)
    from scripts.write_expanded_report import main as write_report
    write_report()
    print(json.dumps(progress),flush=True)


if __name__=='__main__':main()
