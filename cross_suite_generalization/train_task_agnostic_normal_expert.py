#!/usr/bin/env python3
"""Train a task-ID-free normal dynamics ensemble with episode-level folds."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


class Expert(torch.nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(input_dim, 128), torch.nn.SiLU(),
            torch.nn.Linear(128, 128), torch.nn.SiLU(), torch.nn.Linear(128, 6))
    def forward(self, x): return self.net(x)


def nll(raw, target):
    mean, logvar = raw[:, :3], raw[:, 3:].clamp(-6.0, 3.0)
    return .5 * (logvar + (target-mean).square()*torch.exp(-logvar)).sum(1).mean()


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--epochs',type=int,default=400);p.add_argument('--folds',type=int,default=5);a=p.parse_args()
    d=np.load(a.dataset,allow_pickle=False);x=d['x'].astype(np.float64)[:,:-10];y=d['y'].astype(np.float64);eid=d['episode_id']
    if d['abnormal'].any(): raise ValueError('normal dataset contains abnormal rows')
    fold=np.asarray([int(hashlib.sha256(str(e).encode()).hexdigest()[:8],16)%a.folds for e in eid])
    members=[];oof_mean=np.zeros_like(y);oof_var=np.zeros_like(y);summaries=[];torch.set_num_threads(2)
    for f in range(a.folds):
        train=fold!=f;valid=~train;xm=x[train].mean(0);xs=x[train].std(0);xs[xs<1e-6]=1;ym=y[train].mean(0);ys=y[train].std(0);ys[ys<1e-6]=1
        tx=torch.tensor((x[train]-xm)/xs,dtype=torch.float32);ty=torch.tensor((y[train]-ym)/ys,dtype=torch.float32)
        vx=torch.tensor((x[valid]-xm)/xs,dtype=torch.float32);vy=torch.tensor((y[valid]-ym)/ys,dtype=torch.float32)
        seed=20260950+f;torch.manual_seed(seed);net=Expert(x.shape[1]);opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=2e-4);best=None
        for epoch in range(a.epochs):
            net.train();opt.zero_grad();loss=nll(net(tx),ty);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
            if epoch%10==0 or epoch==a.epochs-1:
                net.eval()
                with torch.no_grad():val=float(nll(net(vx),vy))
                if best is None or val<best[0]:best=(val,epoch,copy.deepcopy(net.state_dict()))
        net.load_state_dict(best[2]);net.eval()
        with torch.no_grad():raw=net(vx).numpy()
        oof_mean[valid]=raw[:,:3]*ys+ym;oof_var[valid]=np.exp(np.clip(raw[:,3:],-6,3))*ys**2
        members.append({'state_dict':net.state_dict(),'x_mean':xm,'x_scale':xs,'y_mean':ym,'y_scale':ys,'seed':seed})
        summaries.append({'fold':f,'validation_nll':best[0],'best_epoch':best[1],'train_rows':int(train.sum()),'validation_rows':int(valid.sum())})
    variance_scale=np.clip(np.mean((y-oof_mean)**2/np.maximum(oof_var,1e-12),axis=0),.25,25)
    payload={'model_type':'task_agnostic_normal_ensemble','input_dim':x.shape[1],'members':members,
             'variance_scale_global':variance_scale,'task_identity_used':False,'folds':a.folds}
    a.out.parent.mkdir(parents=True,exist_ok=True);torch.save(payload,a.out);digest=hashlib.sha256(a.out.read_bytes()).hexdigest()
    report={'status':'trained_not_calibrated','rows':len(x),'episodes':len(np.unique(eid)),'folds':summaries,
            'variance_scale_global':variance_scale.tolist(),'checkpoint_sha256':digest,'task_identity_used':False}
    a.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
