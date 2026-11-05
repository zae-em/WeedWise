"""Briefly pause only this project's downloader to let an authorized transfer finish."""
import json
import time
from pathlib import Path
import psutil

ROOT=Path(__file__).resolve().parents[1]


if __name__=='__main__':
    held=[]
    try:
        for process in psutil.process_iter(['name','cmdline']):
            try:
                if (process.info['name'] or '').lower() not in {'python.exe','python','python3'}:continue
                if not any(Path(arg).name=='download_cropandweed.py' for arg in process.info['cmdline'] or []):continue
                if Path(process.cwd()).resolve()!=ROOT:continue
                process.suspend();held.append(process)
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        (ROOT/'evaluation/download_traffic_pause.json').write_text(json.dumps({'paused_pids':[p.pid for p in held],'resume_unix':time.time()+300},indent=2))
        print('Download traffic paused for at most five minutes:',[p.pid for p in held],flush=True)
        time.sleep(300)
    finally:
        for process in held:
            try:process.resume()
            except psutil.NoSuchProcess:pass
        (ROOT/'evaluation/download_traffic_pause.json').write_text(json.dumps({'resumed':True,'resumed_unix':time.time()},indent=2))
        print('Download processes resumed.',flush=True)
