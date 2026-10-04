#!/usr/bin/env python3
"""Score DROID calibration data with conditional dual-expert ensemble."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from conditional_relational_dual_expert import conditional_features,episode_vector,make_model

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-directory',type=Path,required=True);p.add_argument('--model',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if len(a.model)<3:raise ValueError('at least three models required')
 import torch
 from torch import nn
 manifest=json.loads((a.data_directory/'manifest.json').read_text(encoding='utf-8'));path=Path(manifest['output_files']['calibration']['path']);path=path if path.is_absolute() else a.data_directory/path;data=np.load(path);episode,counts=episode_vector(manifest,'calibration');x,y=conditional_features(data,episode,15);action=np.concatenate([np.arange(n,dtype=np.int32) for n in counts])
 members=[]
 for path in a.model:
  c=torch.load(path,map_location='cpu',weights_only=False);cfg=c['config'];m=make_model(torch,nn,cfg['input_dim'],cfg['output_dim'],cfg['seed']);m.load_state_dict(c['state_dict']);m.eval();members.append((c,m))
 def score(target):
  ns=[];ass=[]
  for c,m in members:
   X=torch.from_numpy(((x-c['xcenter'])/c['xscale']).astype(np.float32));Y=torch.from_numpy(((target-c['ycenter'])/c['yscale']).astype(np.float32))
   with torch.no_grad():ns.append(m.logp(X,Y,False).numpy());ass.append(m.logp(X,Y,True).numpy())
  ns=np.stack(ns);ass=np.stack(ass);return ns.mean(0),ass.mean(0),(ass-ns).std(0)
 arrays={k:[] for k in ('normal_logp','abnormal_logp','ensemble_std','abnormal','scale','episode','action','reliability_fit')}
 for scale_index,scale in enumerate([1.]+list(members[0][0]['config']['scales'])):
  n,a_,std=score(y*scale);is_abnormal=scale_index>0;offset=scale_index*len(counts)
  arrays['normal_logp'].append(n);arrays['abnormal_logp'].append(a_);arrays['ensemble_std'].append(std);arrays['abnormal'].append(np.full(len(y),is_abnormal,bool));arrays['scale'].append(np.full(len(y),scale,np.float32));arrays['episode'].append(episode+offset);arrays['action'].append(action);arrays['reliability_fit'].append(episode%2==0)
 a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**{k:np.concatenate(v) for k,v in arrays.items()});print(json.dumps({'models':len(members),'rows':int(4*len(y)),'episodes_per_condition':len(counts)},indent=2))
if __name__=='__main__':main()
