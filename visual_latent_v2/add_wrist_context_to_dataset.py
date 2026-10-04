#!/usr/bin/env python3
"""Attach pre-action wrist appearance context to an aligned response dataset."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import cv2,numpy as np


def context(images):
    rows=[]
    for image in images:
        gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        gray8=cv2.resize(gray,(8,8),interpolation=cv2.INTER_AREA).astype(np.float32)/255
        color4=cv2.resize(image,(4,4),interpolation=cv2.INTER_AREA).astype(np.float32)/255
        rows.append(np.r_[gray8.ravel(),color4.ravel()])
    return np.asarray(rows,np.float32)


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--visual-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    d=np.load(a.dataset,allow_pickle=False);sid=d['sample_id'].astype(str);action=d['action_index'];out=np.zeros((len(sid),112),np.float32)
    for sample in np.unique(sid):
        ix=np.flatnonzero(sid==sample);ix=ix[np.argsort(action[ix])];files=list((a.visual_root/sample).glob('*.npz'))
        if len(files)!=1:raise ValueError(f'{sample}: expected one visual file, got {files}')
        raw=np.load(files[0],allow_pickle=False)
        if not np.array_equal(action[ix],raw['action_indices']):raise ValueError(f'{sample}: action mismatch')
        out[ix]=context(raw['wrist_images'])
    arrays={k:d[k] for k in d.files if k!='metadata_json'};meta=json.loads(str(d['metadata_json']));meta['wrist_context']={'dim':112,'timing':'current pre-action image','features':'gray8x8 plus RGB4x4','uses_future':False};arrays['wrist_context']=out;arrays['metadata_json']=np.asarray(json.dumps(meta,ensure_ascii=False));np.savez_compressed(a.out,**arrays)
    print(json.dumps({'rows':len(out),'context_dim':out.shape[1],'samples':len(np.unique(sid))}))
if __name__=='__main__':main()
