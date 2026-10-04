#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from libero_droid_transfer import load_trace
from relational_dual_expert import make_model, relational_features

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--trace',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 import torch
 from torch import nn
 c=torch.load(a.model,map_location='cpu',weights_only=False);cfg=c['config'];dummy=np.zeros((cfg['components'],c['summary']['feature_dim']),np.float32)
 model=make_model(torch,nn,cfg['components'],c['summary']['feature_dim'],cfg['seed'],dummy,dummy);model.load_state_dict(c['state_dict']);model.eval()
 arrays={k:[] for k in ('score','abnormal','task_id','episode_idx','action_index','file_index')};reports=[]
 for fi,path in enumerate(a.trace):
  data,meta=load_trace(path);f=relational_features(data['command_history_cartesian_velocity'],data['response_eef_delta_xyz_euler'],20,cfg.get('feature_set','full'));f=(f-c['center'])/c['scale']
  with torch.no_grad(): x=torch.from_numpy(f);score=(model.logp(x,True)-model.logp(x,False)).numpy()
  for k in ('abnormal','task_id','episode_idx','action_index'):arrays[k].append(meta[k])
  arrays['score'].append(score);arrays['file_index'].append(np.full(len(score),fi,np.int32));reports.append({'path':str(path),'rows':len(score)})
 np.savez_compressed(a.output,**{k:np.concatenate(v) for k,v in arrays.items()});print(json.dumps({'files':len(reports),'rows':sum(x['rows'] for x in reports)},indent=2))
if __name__=='__main__':main()
