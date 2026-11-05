"""Transfer authorized inputs and start the expansion on the verified selected T4."""
import sys
import argparse
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_expansion_colab import prepare
from scripts.watch_colab_job import request
from scripts.notebook_transport import upload,execute


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transport',choices=['auto','notebook','filesystem'],default='auto')
    args=parser.parse_args()
    files,source=prepare()
    print('Transferring hashed original data and v1 weights through the official Colab filesystem...',flush=True)
    transferred=False
    if args.transport!='notebook':
        try:
            request('upload',timeout=1800,files=files)
            print('Official upload returned; hashes are verified in the notebook.',flush=True)
            transferred=True
        except RuntimeError as error:
            if args.transport=='filesystem':raise
            print('Filesystem transfer unavailable:',error,'Using public notebook execution.',flush=True)
    if not transferred:
        for index,file in enumerate(files):
            print('Uploading through notebook',index+1,'/',len(files),file,flush=True)
            upload(ROOT/file)
    print(execute(source,timeout=900),flush=True)
