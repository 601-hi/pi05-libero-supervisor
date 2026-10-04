#!/usr/bin/env python3
"""Export per-transition source-domain calibration scores for frozen thresholds."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from libero_droid_transfer import INPUT_KEYS, TARGET_KEYS, build_model, continuous_score, normalize


def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--readiness',type=Path,required=True); p.add_argument('--data-directory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    import torch
    from torch import nn
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    readiness=json.loads(a.readiness.read_text(encoding='utf-8'))
    manifest=json.loads((a.data_directory/'manifest.json').read_text(encoding='utf-8'))
    data=np.load(a.data_directory/manifest['output_files']['calibration']['path'],allow_pickle=False)
    x=normalize(data,INPUT_KEYS,readiness['normalization']); y=normalize(data,TARGET_KEYS,readiness['normalization'])
    model=build_model(torch,nn,checkpoint,x.shape[1],y.shape[1]); score=continuous_score(model,torch,x,y)
    # DROID bundle does not expose episode identity as a feature. Each transition is
    # treated independently for source calibration; smoothing is therefore disabled
    # by assigning a distinct episode id to each row.
    n=len(score); ids=np.arange(n,dtype=np.int64)
    np.savez_compressed(a.output,score=score,abnormal=np.zeros(n,bool),task_id=np.zeros(n,np.int32),
                        episode_idx=ids,action_index=np.zeros(n,np.int32),file_index=np.zeros(n,np.int32))
    print(json.dumps({'rows':n,'p95':float(np.percentile(score,95)),'p99':float(np.percentile(score,99))},indent=2))
if __name__=='__main__': main()
