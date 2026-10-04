#!/usr/bin/env python3
"""Score task-ID-free normal/abnormal response density experts."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np,torch
from train_task_agnostic_normal_expert import Expert as NormalExpert
from train_task_agnostic_abnormal_expert import Expert as AbnormalExpert
LOG2PI=float(np.log(2*np.pi))
def normal_logp(c,x,y):
 values=[]
 for m in c['members']:
  net=NormalExpert(c['input_dim']);net.load_state_dict(m['state_dict']);net.eval();xm=np.asarray(m['x_mean']);xs=np.asarray(m['x_scale']);ym=np.asarray(m['y_mean']);ys=np.asarray(m['y_scale'])
  with torch.no_grad():r=net(torch.tensor((x-xm)/xs,dtype=torch.float32)).numpy()
  mean=r[:,:3]*ys+ym;var=np.exp(np.clip(r[:,3:],-6,3))*ys**2*np.asarray(c['variance_scale_global'])
  values.append(-.5*np.sum(np.log(var)+(y-mean)**2/var+LOG2PI,axis=1))
 values=np.asarray(values);m=values.max(0);return m+np.log(np.exp(values-m).mean(0))
def abnormal_logp(c,x,y):
 net=AbnormalExpert(c['input_dim'],c['components']);net.load_state_dict(c['state_dict']);net.eval();xm=np.asarray(c['x_mean']);xs=np.asarray(c['x_scale']);ym=np.asarray(c['y_mean']);ys=np.asarray(c['y_scale'])
 with torch.no_grad():logits,mean,logvar=net(torch.tensor((x-xm)/xs,dtype=torch.float32))
 logits,mean,logvar=logits.numpy(),mean.numpy(),logvar.numpy();target=(y-ym)/ys;component=-.5*np.sum(logvar+(target[:,None,:]-mean)**2/np.exp(logvar)+LOG2PI,axis=2);weight=logits-np.logaddexp.reduce(logits,axis=1)[:,None];joint=weight+component;m=joint.max(1);return m+np.log(np.exp(joint-m[:,None]).sum(1))-np.log(ys).sum()
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--normal',type=Path,required=True);p.add_argument('--abnormal-expert',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 d=np.load(a.dataset,allow_pickle=False);x=d['x'].astype(np.float64)[:,:-10];y=d['y'].astype(np.float64);n=torch.load(a.normal,map_location='cpu',weights_only=False);ab=torch.load(a.abnormal_expert,map_location='cpu',weights_only=False);torch.set_num_threads(2);nl=normal_logp(n,x,y);al=abnormal_logp(ab,x,y)
 np.savez_compressed(a.out,normal_logp=nl,abnormal_logp=al,abnormal=d['abnormal'],task_id=d['task_id'],episode_id=d['episode_id'],action_index=d['action_index'],scale=d['scale'],onset_mode=d['onset_mode']);print({'rows':len(x),'finite':bool(np.isfinite(nl).all() and np.isfinite(al).all())})
if __name__=='__main__':main()
