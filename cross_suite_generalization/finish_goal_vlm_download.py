"""Finish this run's HF partial with verified parallel ranges, without duplicates.

The standard downloader MUST be stopped first. A sparse work suffix prevents
an interrupted ranged file from being mistaken for a valid contiguous prefix.
Only a complete SHA256-verified blob is installed into the HF cache.
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import requests
from huggingface_hub import HfApi, hf_hub_url, get_hf_file_metadata, snapshot_download


def main():
    repo='Qwen/Qwen3-VL-2B-Instruct'; rev='89644892e4d85e24eaac8bacfd4f463576704203'
    info=HfApi().model_info(repo,revision=rev,files_metadata=True,timeout=20)
    spec=next(s for s in info.siblings if s.rfilename=='model.safetensors')
    sha=spec.lfs.sha256; size=spec.size
    cache=Path('/root/gpufree-data/hf-cache/hub/models--Qwen--Qwen3-VL-2B-Instruct/blobs')
    partial=cache/(sha+'.incomplete'); work=cache/(sha+'.ranged_work'); final=cache/sha
    if final.exists() or work.exists():
        raise RuntimeError('Existing final or ranged work requires inspection; refusing overwrite')
    prefix=partial.stat().st_size if partial.exists() else 0
    if not 0 <= prefix < size or shutil.disk_usage(cache).free-(size-prefix)<3*1024**3:
        raise RuntimeError('invalid prefix or insufficient 3 GiB reserve')
    meta=get_hf_file_metadata(hf_hub_url(repo,'model.safetensors',revision=rev),timeout=20)
    if meta.etag != sha or meta.size != size:
        raise RuntimeError('metadata checksum/size disagreement')
    with requests.get(meta.location,headers={'Range':'bytes=0-0'},stream=True,timeout=30) as probe:
        if probe.status_code!=206 or probe.headers.get('Content-Range')!=f'bytes 0-0/{size}':
            raise RuntimeError('server does not honor verified ranges')
    if partial.exists():
        partial.rename(work)
    else:
        work.touch(exist_ok=False)
    descriptor=os.open(work,os.O_RDWR)
    edges=[prefix+(size-prefix)*i//6 for i in range(7)]
    def fetch(start,end):
        with requests.get(meta.location,headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(20,60)) as response:
            if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':
                raise RuntimeError('incorrect range response')
            offset=start
            for block in response.iter_content(1024*1024):
                if offset+len(block)>end+1: raise RuntimeError('oversized range')
                view=memoryview(block)
                while view:
                    written=os.pwrite(descriptor,view,offset)
                    if written<=0: raise OSError('short positional write')
                    offset+=written; view=view[written:]
            if offset!=end+1: raise RuntimeError('truncated range')
        print(f'completed_range={start}-{end}',flush=True)
    print(json.dumps(dict(prefix_bytes=prefix,total_bytes=size,workers=6,sha256=sha)),flush=True)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            futures=[pool.submit(fetch,edges[i],edges[i+1]-1) for i in range(6)]
            for future in concurrent.futures.as_completed(futures): future.result()
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    digest=hashlib.sha256()
    with work.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''): digest.update(block)
    if work.stat().st_size!=size or digest.hexdigest()!=sha:
        raise RuntimeError('SHA256 verification failed; ranged file retained for inspection')
    work.rename(final)
    path=snapshot_download(repo,revision=rev,local_files_only=True)
    snapshot=Path(path)
    weight_link=snapshot/'model.safetensors'
    if not weight_link.exists(): weight_link.symlink_to('../../blobs/'+sha)
    result=dict(repo=repo,revision=rev,snapshot=path,downloaded=True,weight_sha256=sha,
        weight_bytes=size,verification='LFS SHA256 verified after parallel ranges')
    Path('../GOAL_VLM_MODEL_20260920.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
