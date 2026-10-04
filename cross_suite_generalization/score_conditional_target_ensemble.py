#!/usr/bin/env python3
"""Zero-fit target scoring for frozen conditional relational ensemble."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from conditional_relational_dual_expert import conditional_features,make_model
from libero_droid_transfer import load_trace
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,action='append',required=True);p.add_argument('--trace',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();
 if len(a.model)<3:raise ValueError('at least three models required')
 import torch
 from torch import nn
 members=[]
 for path in a.model:
  c=torch.load(path,map_location='cpu',weights_only=False);cfg=c['config'];m=make_model(torch,nn,cfg['input_dim'],cfg['output_dim'],cfg['seed']);m.load_state_dict(c['state_dict']);m.eval();members.append((c,m))
 arrays={k:[] for k in ('normal_logp','abnormal_logp','ensemble_std','abnormal','task_id','episode_idx','action_index','file_index')}
 for fi,path in enumerate(a.trace):
  data,meta=load_trace(path);episode=np.asarray(meta['episode_idx'],np.int32);x,y=conditional_features(data,episode,20,gripper_scale=1.0);ns=[];ass=[]
  for c,m in members:
   X=torch.from_numpy(((x-c['xcenter'])/c['xscale']).astype(np.float32));Y=torch.from_numpy(((y-c['ycenter'])/c['yscale']).astype(np.float32))
   with torch.no_grad():ns.append(m.logp(X,Y,False).numpy());ass.append(m.logp(X,Y,True).numpy())
  ns=np.stack(ns);ass=np.stack(ass);arrays['normal_logp'].append(ns.mean(0));arrays['abnormal_logp'].append(ass.mean(0));arrays['ensemble_std'].append((ass-ns).std(0))
  for key in ('abnormal','task_id','episode_idx','action_index'):arrays[key].append(meta[key])
  arrays['file_index'].append(np.full(len(y),fi,np.int32))
 a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**{k:np.concatenate(v) for k,v in arrays.items()});print(json.dumps({'models':len(members),'files':len(a.trace),'rows':int(sum(map(len,arrays['normal_logp'])))},indent=2))
if __name__=='__main__':main()
