#!/usr/bin/env python3
"""Source-only pilot evaluation for a conditional relational dual expert."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from conditional_relational_dual_expert import conditional_features,episode_vector,make_model

def auc(values,labels):
 values=np.asarray(values);labels=np.asarray(labels,bool);order=np.argsort(values);ranks=np.empty(len(values));ranks[order]=np.arange(1,len(values)+1);n1=labels.sum();n0=(~labels).sum();return float((ranks[labels].sum()-n1*(n1+1)/2)/(n1*n0))
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-directory',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 import torch
 from torch import nn
 manifest=json.loads((a.data_directory/'manifest.json').read_text(encoding='utf-8'));path=Path(manifest['output_files']['calibration']['path']);path=path if path.is_absolute() else a.data_directory/path;data=np.load(path);episode,_=episode_vector(manifest,'calibration');x,y=conditional_features(data,episode,15)
 c=torch.load(a.model,map_location='cpu',weights_only=False);cfg=c['config'];model=make_model(torch,nn,cfg['input_dim'],cfg['output_dim'],cfg['seed']);model.load_state_dict(c['state_dict']);model.eval();X=torch.from_numpy(((x-c['xcenter'])/c['xscale']).astype(np.float32))
 def scores(target):
  Y=torch.from_numpy(((target-c['ycenter'])/c['yscale']).astype(np.float32))
  with torch.no_grad():return (model.logp(X,Y,True)-model.logp(X,Y,False)).numpy()
 normal=scores(y);by_scale={};all_a=[]
 for scale in cfg['scales']:
  value=scores(y*scale);all_a.append(value);by_scale[str(scale)]={'auc':auc(np.r_[normal,value],np.r_[np.zeros(len(normal),bool),np.ones(len(value),bool)]),'normal_median':float(np.median(normal)),'abnormal_median':float(np.median(value))}
 combined=np.concatenate(all_a);result={'source_only':True,'model_summary':c['summary'],'combined_auc':auc(np.r_[normal,combined],np.r_[np.zeros(len(normal),bool),np.ones(len(combined),bool)]),'by_scale':by_scale}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
