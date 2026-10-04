#!/usr/bin/env python3
"""Score DROID calibration episodes with a frozen relational dual expert."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from relational_dual_expert import make_model,relational_features

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--data-directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 import torch
 from torch import nn
 c=torch.load(a.model,map_location='cpu',weights_only=False);cfg=c['config'];manifest=json.loads((a.data_directory/'manifest.json').read_text(encoding='utf-8'))
 data=np.load(a.data_directory/manifest['output_files']['calibration']['path'],allow_pickle=False)
 f=relational_features(data['command_history_cartesian_velocity'],data['response_eef_delta_xyz_euler'],15,cfg['feature_set']);f=(f-c['center'])/c['scale']
 dummy=np.zeros((cfg['components'],c['summary']['feature_dim']),np.float32);model=make_model(torch,nn,cfg['components'],c['summary']['feature_dim'],cfg['seed'],dummy,dummy);model.load_state_dict(c['state_dict']);model.eval()
 with torch.no_grad():x=torch.from_numpy(f);score=(model.logp(x,True)-model.logp(x,False)).numpy()
 counts=[int(row['exported_transitions']) for row in manifest['episodes'] if row['split']=='calibration' and row['accepted']]
 if sum(counts)!=len(score):raise ValueError(f'episode counts {sum(counts)} != scores {len(score)}')
 episode=np.concatenate([np.full(count,i,np.int32) for i,count in enumerate(counts)]);action=np.concatenate([np.arange(count,dtype=np.int32) for count in counts]);n=len(score)
 np.savez_compressed(a.output,score=score,abnormal=np.zeros(n,bool),task_id=np.zeros(n,np.int32),episode_idx=episode,action_index=action,file_index=np.zeros(n,np.int32))
 print(json.dumps({'rows':n,'episodes':len(counts),'single_q95':float(np.percentile(score,95)),'single_q99':float(np.percentile(score,99))},indent=2))
if __name__=='__main__':main()
