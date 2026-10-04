#!/usr/bin/env python3
"""Cross-robot conditional dual experts over dimensionless directed progress."""
from __future__ import annotations
import argparse,copy,json
from pathlib import Path
import numpy as np

try:
    from .relational_dual_expert import relational_features, robust_fit
except ImportError:  # Preserve direct-script training entry points.
    from relational_dual_expert import relational_features, robust_fit

LOW=np.asarray([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-.0175,-2.8973],np.float32)
HIGH=np.asarray([2.8973,1.7628,2.8973,-.0698,2.8973,3.7525,2.8973],np.float32)
REACH_M=.855

def episode_vector(manifest,split):
 counts=[int(r['exported_transitions']) for r in manifest['episodes'] if r['split']==split and r['accepted']]
 return np.concatenate([np.full(n,i,np.int32) for i,n in enumerate(counts)]),counts

def conditional_features(data,episode,frequency_hz,gripper_scale=1.0):
 cmd=np.asarray(data['command_history_cartesian_velocity'],np.float32)/frequency_hz
 current=cmd[:,0];trans=current[:,:3];norm=np.linalg.norm(trans,axis=1,keepdims=True);unit=trans/(norm+1e-8)
 q=np.asarray(data['state_joint_position'],np.float32);qnorm=2*(q-LOW)/(HIGH-LOW)-1
 margin=np.min(np.minimum((q-LOW)/(HIGH-LOW),(HIGH-q)/(HIGH-LOW)),axis=1,keepdims=True)
 eef=np.asarray(data['state_eef_pose_xyz_euler'],np.float32)[:,:3];initial=np.empty_like(eef)
 for ep in np.unique(episode):initial[episode==ep]=eef[np.flatnonzero(episode==ep)[0]]
 relative=(eef-initial)/REACH_M
 grip=np.clip(np.asarray(data['state_gripper_position'],np.float32)[:,:1]/gripper_scale,0,1)
 response=np.asarray(data['response_eef_delta_xyz_euler'],np.float32)[:,:3]
 progress=relational_features(data['command_history_cartesian_velocity'],data['response_eef_delta_xyz_euler'],frequency_hz,'progress_only')
 previous_response=np.zeros_like(response);previous_progress=np.zeros((len(response),1),np.float32)
 for ep in np.unique(episode):
  idx=np.flatnonzero(episode==ep);previous_response[idx[1:]]=response[idx[:-1]]/REACH_M;previous_progress[idx[1:],0]=progress[idx[:-1],0]
 rotation_norm=np.linalg.norm(current[:,3:6],axis=1,keepdims=True)/np.pi
 context=np.c_[qnorm,margin,relative,grip,unit,norm/REACH_M,rotation_norm,previous_response,previous_progress]
 return context.astype(np.float32),progress.astype(np.float32)

def make_model(torch,nn,input_dim,output_dim,seed):
 torch.manual_seed(seed)
 class ConditionalDual(nn.Module):
  def __init__(self):
   super().__init__()
   def expert():return nn.Sequential(nn.Linear(input_dim,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,2*output_dim))
   self.normal=expert();self.abnormal=expert()
  def logp(self,x,y,abnormal=False):
   out=(self.abnormal if abnormal else self.normal)(x);mean,logvar=out[:,:output_dim],out[:,output_dim:].clamp(-6,4)
   return -.5*(logvar+(y-mean).square()*torch.exp(-logvar)+np.log(2*np.pi)).sum(1)
 return ConditionalDual()

def train(args):
 import torch
 from torch import nn
 torch.set_num_threads(args.threads);torch.manual_seed(args.seed)
 manifest=json.loads((args.data_directory/'manifest.json').read_text(encoding='utf-8'))
 def load(split):
  path=Path(manifest['output_files'][split]['path']);path=path if path.is_absolute() else args.data_directory/path
  data=np.load(path);episode,_=episode_vector(manifest,split);x,y=conditional_features(data,episode,15);return x,y,episode
 tx,ty,_=load('train');cx,cy,ce=load('calibration');xcenter,xscale=robust_fit(tx);ycenter,yscale=robust_fit(ty)
 tx=(tx-xcenter)/xscale;cx=(cx-xcenter)/xscale;ty=(ty-ycenter)/yscale;cy=(cy-ycenter)/yscale
 scales=np.asarray(args.scales,np.float32);model=make_model(torch,nn,tx.shape[1],ty.shape[1],args.seed);opt=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-4)
 X=torch.from_numpy(tx);Y=torch.from_numpy(ty);rng=torch.Generator().manual_seed(args.seed);best=None;best_loss=float('inf');stale=0;history=[]
 for epoch in range(args.epochs):
  model.train();order=torch.randperm(len(X),generator=rng)
  for start in range(0,len(X),args.batch_size):
   idx=order[start:start+args.batch_size];x=X[idx];yn=Y[idx];chosen=torch.from_numpy(scales)[torch.randint(len(scales),(len(idx),),generator=rng)];ya=(yn*torch.from_numpy(yscale)+torch.from_numpy(ycenter))*chosen[:,None];ya=(ya-torch.from_numpy(ycenter))/torch.from_numpy(yscale)
   nnlp=model.logp(x,yn,False);nalp=model.logp(x,yn,True);aalp=model.logp(x,ya,True);anlp=model.logp(x,ya,False)
   loss=-nnlp.mean()-aalp.mean()+args.margin_weight*(torch.relu(args.margin-(aalp-anlp)).mean()+torch.relu(args.margin+(nalp-nnlp)).mean())
   opt.zero_grad();loss.backward();opt.step()
  model.eval()
  with torch.no_grad():
   vx=torch.from_numpy(cx);vyn=torch.from_numpy(cy);vals=[]
   for scale in scales:
    vya=(vyn*torch.from_numpy(yscale)+torch.from_numpy(ycenter))*scale;vya=(vya-torch.from_numpy(ycenter))/torch.from_numpy(yscale)
    nnlp=model.logp(vx,vyn,False);nalp=model.logp(vx,vyn,True);aalp=model.logp(vx,vya,True);anlp=model.logp(vx,vya,False)
    vals.append(-nnlp.mean()-aalp.mean()+args.margin_weight*(torch.relu(args.margin-(aalp-anlp)).mean()+torch.relu(args.margin+(nalp-nnlp)).mean()))
   val=float(torch.stack(vals).mean());history.append({'epoch':epoch+1,'calibration_objective':val})
  if val<best_loss-1e-4:best_loss=val;best=copy.deepcopy(model.state_dict());best_epoch=epoch+1;stale=0
  else:
   stale+=1
   if stale>=args.patience:break
 model.load_state_dict(best);payload={'state_dict':model.state_dict(),'xcenter':xcenter,'xscale':xscale,'ycenter':ycenter,'yscale':yscale,'config':{'seed':args.seed,'input_dim':tx.shape[1],'output_dim':ty.shape[1],'scales':args.scales,'margin':args.margin,'margin_weight':args.margin_weight},'summary':{'best_epoch':best_epoch,'epochs':len(history),'best_calibration_objective':best_loss,'train_rows':len(tx),'calibration_rows':len(cx)}}
 args.output.parent.mkdir(parents=True,exist_ok=True);torch.save(payload,args.output);args.output.with_suffix('.history.json').write_text(json.dumps(history,indent=2)+'\n',encoding='utf-8');print(json.dumps(payload['summary'],indent=2))

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=202609121);p.add_argument('--threads',type=int,default=2);p.add_argument('--batch-size',type=int,default=1024);p.add_argument('--epochs',type=int,default=150);p.add_argument('--patience',type=int,default=15);p.add_argument('--learning-rate',type=float,default=1e-3);p.add_argument('--margin',type=float,default=2.);p.add_argument('--margin-weight',type=float,default=1.);p.add_argument('--scales',type=float,nargs='+',default=[.25,.5,.75]);train(p.parse_args())
if __name__=='__main__':main()
