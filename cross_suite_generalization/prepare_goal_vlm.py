"""Download one pinned official small VLM after checking free disk capacity."""
import argparse
import json
from pathlib import Path
import shutil
from huggingface_hub import HfApi, snapshot_download


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--download',action='store_true')
    args=p.parse_args()
    repo='Qwen/Qwen3-VL-2B-Instruct'
    info=HfApi().model_info(repo,files_metadata=True,timeout=20)
    files=[s for s in info.siblings if s.rfilename.endswith(('.json','.safetensors','.txt','.model'))]
    required=sum(s.size or 0 for s in files)
    free=shutil.disk_usage('/root/gpufree-data').free
    result=dict(repo=repo,revision=info.sha,required_bytes=required,free_bytes=free,
        files=[s.rfilename for s in files],downloaded=False)
    print(json.dumps(result),flush=True)
    if args.download:
        if required<=0 or free-required<3*1024**3:
            raise RuntimeError('Insufficient room with 3 GiB reserve; no existing files will be deleted')
        result['snapshot']=snapshot_download(repo,revision=info.sha,
            allow_patterns=result['files'],max_workers=2)
        result['downloaded']=True
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
