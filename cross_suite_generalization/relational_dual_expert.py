#!/usr/bin/env python3
"""Source-trained dual density experts over dimensionless command-response relations."""
from __future__ import annotations
import argparse, copy, hashlib, json
from pathlib import Path
import numpy as np


def relational_features(command_velocity, response, frequency_hz, feature_set="full"):
    command_displacement=np.asarray(command_velocity,dtype=np.float32)/float(frequency_hz)
    response=np.asarray(response,dtype=np.float32)[:,:3]
    rnorm=np.linalg.norm(response,axis=1)
    pieces=[]
    for lag in range(command_displacement.shape[1]):
        command=command_displacement[:,lag,:3]; cnorm=np.linalg.norm(command,axis=1)
        dot=np.sum(command*response,axis=1)
        progress=dot/(cnorm*cnorm+1e-10)
        cosine=dot/(cnorm*rnorm+1e-10)
        log_ratio=np.log((rnorm+1e-4)/(cnorm+1e-4))
        if feature_set == "progress_only": pieces.append(np.clip(progress,-2,2))
        elif feature_set == "full": pieces.extend((np.clip(progress,-2,2),np.clip(cosine,-1,1),np.clip(log_ratio,-5,5)))
        else: raise ValueError(f"unknown feature_set: {feature_set}")
    return np.stack(pieces,axis=1).astype(np.float32)


def robust_fit(x):
    center=np.median(x,axis=0); scale=np.percentile(x,75,axis=0)-np.percentile(x,25,axis=0)
    scale=np.where(scale>1e-6,scale,1.0)
    return center.astype(np.float32),scale.astype(np.float32)


def make_model(torch, nn, components, dimension, seed, normal_init, abnormal_init):
    class DualGMM(nn.Module):
        def __init__(self):
            super().__init__(); generator=torch.Generator().manual_seed(seed)
            self.n_logits=nn.Parameter(torch.zeros(components)); self.a_logits=nn.Parameter(torch.zeros(components))
            ni=torch.randperm(len(normal_init),generator=generator)[:components]
            ai=torch.randperm(len(abnormal_init),generator=generator)[:components]
            self.n_mean=nn.Parameter(torch.from_numpy(normal_init[ni.numpy()].copy()))
            self.a_mean=nn.Parameter(torch.from_numpy(abnormal_init[ai.numpy()].copy()))
            self.n_logvar=nn.Parameter(torch.zeros(components,dimension))
            self.a_logvar=nn.Parameter(torch.zeros(components,dimension))
        def logp(self,x,abnormal=False):
            logits=self.a_logits if abnormal else self.n_logits; mean=self.a_mean if abnormal else self.n_mean
            logvar=(self.a_logvar if abnormal else self.n_logvar).clamp(-6,4)
            component=-.5*(logvar+(x[:,None,:]-mean)**2*torch.exp(-logvar)+np.log(2*np.pi)).sum(2)
            return torch.logsumexp(torch.log_softmax(logits,0)[None,:]+component,1)
    return DualGMM()


def train(args):
    import torch
    from torch import nn
    torch.manual_seed(args.seed); torch.set_num_threads(args.threads)
    manifest=json.loads((args.data_directory/'manifest.json').read_text(encoding='utf-8'))
    train=np.load(args.data_directory/manifest['output_files']['train']['path'],allow_pickle=False)
    cal=np.load(args.data_directory/manifest['output_files']['calibration']['path'],allow_pickle=False)
    normal=relational_features(train['command_history_cartesian_velocity'],train['response_eef_delta_xyz_euler'],15,args.feature_set)
    cal_normal=relational_features(cal['command_history_cartesian_velocity'],cal['response_eef_delta_xyz_euler'],15,args.feature_set)
    abnormal=np.concatenate([relational_features(train['command_history_cartesian_velocity'],
        train['response_eef_delta_xyz_euler']*s,15,args.feature_set) for s in args.scales])
    cal_abnormal=np.concatenate([relational_features(cal['command_history_cartesian_velocity'],
        cal['response_eef_delta_xyz_euler']*s,15,args.feature_set) for s in args.scales])
    center,scale=robust_fit(normal); normal=(normal-center)/scale; abnormal=(abnormal-center)/scale
    cal_normal=(cal_normal-center)/scale; cal_abnormal=(cal_abnormal-center)/scale
    model=make_model(torch,nn,args.components,normal.shape[1],args.seed,normal,abnormal).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-4)
    normal_t=torch.from_numpy(normal).to(args.device); abnormal_t=torch.from_numpy(abnormal).to(args.device)
    cn=torch.from_numpy(cal_normal).to(args.device); ca=torch.from_numpy(cal_abnormal).to(args.device)
    rng=torch.Generator().manual_seed(args.seed); history=[]; best=None; best_loss=float('inf'); stale=0
    for epoch in range(args.epochs):
        model.train()
        steps=max((len(normal_t)+args.batch_size-1)//args.batch_size,(len(abnormal_t)+args.batch_size-1)//args.batch_size)
        for step in range(steps):
            ni=torch.randint(len(normal_t),(min(args.batch_size,len(normal_t)),),generator=rng).to(args.device)
            ai=torch.randint(len(abnormal_t),(min(args.batch_size,len(abnormal_t)),),generator=rng).to(args.device)
            n=normal_t[ni]; a=abnormal_t[ai]; nnlp=model.logp(n,False); nalp=model.logp(n,True); aalp=model.logp(a,True); anlp=model.logp(a,False)
            density=-nnlp.mean()-aalp.mean()
            margin=torch.relu(args.margin-(aalp-anlp)).mean()+torch.relu(args.margin+(nalp-nnlp)).mean()
            loss=density+args.margin_weight*margin; optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            cn_n=model.logp(cn,False); cn_a=model.logp(cn,True); ca_a=model.logp(ca,True); ca_n=model.logp(ca,False)
            val=float((-cn_n.mean()-ca_a.mean()+args.margin_weight*(torch.relu(args.margin-(ca_a-ca_n)).mean()+torch.relu(args.margin+(cn_a-cn_n)).mean())).cpu())
            separation={'normal_lr_median':float(torch.median(cn_a-cn_n).cpu()),'abnormal_lr_median':float(torch.median(ca_a-ca_n).cpu())}
        history.append({'epoch':epoch+1,'calibration_objective':val,**separation})
        if val<best_loss-1e-4: best_loss=val; best=copy.deepcopy(model.state_dict()); best_epoch=epoch+1; stale=0
        else:
            stale+=1
            if stale>=args.patience: break
    model.load_state_dict(best)
    saved_config=vars(args).copy()
    payload={'state_dict':model.state_dict(),'center':center,'scale':scale,'config':saved_config,
             'summary':{'best_epoch':best_epoch,'epochs_completed':len(history),'best_calibration_objective':best_loss,
                        'feature_dim':normal.shape[1],'train_normal':len(normal),'train_synthetic_abnormal':len(abnormal)}}
    payload['config']['data_directory']=str(payload['config']['data_directory']); payload['config']['output']=str(payload['config']['output']); payload['config']['device']='cpu'
    args.output.parent.mkdir(parents=True,exist_ok=True); torch.save(payload,args.output)
    args.output.with_suffix('.history.json').write_text(json.dumps(history,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(payload['summary'],indent=2))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-directory',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--components',type=int,default=8);p.add_argument('--seed',type=int,default=20260911);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=2048);p.add_argument('--epochs',type=int,default=100);p.add_argument('--patience',type=int,default=10)
    p.add_argument('--learning-rate',type=float,default=1e-3);p.add_argument('--margin',type=float,default=2.0);p.add_argument('--margin-weight',type=float,default=.1)
    p.add_argument('--scales',type=float,nargs='+',default=[.25,.5,.75]);p.add_argument('--feature-set',choices=('full','progress_only'),default='full')
    p.add_argument('--device',default='cpu'); args=p.parse_args(); train(args)
if __name__=='__main__':main()
