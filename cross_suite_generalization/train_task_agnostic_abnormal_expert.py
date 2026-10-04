#!/usr/bin/env python3
"""Train an abnormal conditional density after removing task identity."""
from __future__ import annotations
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np,torch

class Expert(torch.nn.Module):
 def __init__(self,d,k=3):
  super().__init__();self.k=k;self.net=torch.nn.Sequential(torch.nn.Linear(d,128),torch.nn.SiLU(),torch.nn.Linear(128,128),torch.nn.SiLU(),torch.nn.Linear(128,k*7))
 def forward(self,x):
  r=self.net(x).reshape(-1,self.k,7);return r[...,0],r[...,1:4],r[...,4:7].clamp(-7,3)
def nll(logits,mean,logvar,target):
 r=target[:,None,:]-mean;c=-.5*(logvar+r.square()*torch.exp(-logvar)).sum(-1)-1.5*np.log(2*np.pi)
 return -torch.logsumexp(torch.log_softmax(logits,-1)+c,-1).mean()
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--epochs',type=int,default=600);a=p.parse_args()
 d=np.load(a.dataset,allow_pickle=False);sel=d['abnormal'].astype(bool);x=d['x'][sel].astype(np.float64)[:,:-10];y=d['y'][sel].astype(np.float64);eid=d['episode_id'][sel]
 episodes=np.unique(eid);rng=np.random.default_rng(20260905);rng.shuffle(episodes);ve=set(episodes[:max(1,len(episodes)//5)]);valid=np.asarray([e in ve for e in eid]);train=~valid
 xm=x[train].mean(0);xs=x[train].std(0);xs[xs<1e-6]=1;ym=y[train].mean(0);ys=y[train].std(0);ys[ys<1e-6]=1
 tx=torch.tensor((x[train]-xm)/xs,dtype=torch.float32);ty=torch.tensor((y[train]-ym)/ys,dtype=torch.float32);vx=torch.tensor((x[valid]-xm)/xs,dtype=torch.float32);vy=torch.tensor((y[valid]-ym)/ys,dtype=torch.float32)
 torch.manual_seed(20260905);torch.set_num_threads(2);net=Expert(x.shape[1]);opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=2e-4);best=None
 for epoch in range(a.epochs):
  net.train();opt.zero_grad();loss=nll(*net(tx),ty);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
  if epoch%10==0 or epoch==a.epochs-1:
   net.eval()
   with torch.no_grad():value=float(nll(*net(vx),vy))
   if best is None or value<best[0]:best=(value,epoch,copy.deepcopy(net.state_dict()))
 net.load_state_dict(best[2]);payload={'model_type':'task_agnostic_abnormal_mixture','input_dim':x.shape[1],'components':3,'state_dict':net.state_dict(),'x_mean':xm,'x_scale':xs,'y_mean':ym,'y_scale':ys,'task_identity_used':False}
 a.out.parent.mkdir(parents=True,exist_ok=True);torch.save(payload,a.out);digest=hashlib.sha256(a.out.read_bytes()).hexdigest();report={'status':'trained_not_calibrated','active_rows':len(x),'active_episodes':len(episodes),'best_validation_nll':best[0],'best_epoch':best[1],'checkpoint_sha256':digest,'task_identity_used':False}
 a.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
