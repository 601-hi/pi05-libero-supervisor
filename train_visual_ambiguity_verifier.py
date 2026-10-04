#!/usr/bin/env python3
"""Train a lightweight causal visual verifier only for trajectory-overlap rows."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from two_expert_fusion import TwoExpertFusion


class VisualVerifier(torch.nn.Module):
    def __init__(self, dim):
        super().__init__(); self.net=torch.nn.Sequential(
            torch.nn.Linear(dim,128),torch.nn.SiLU(),torch.nn.Dropout(.1),
            torch.nn.Linear(128,64),torch.nn.SiLU(),torch.nn.Linear(64,1))
    def forward(self,x): return self.net(x).squeeze(-1)


def states(scores, config):
    out=[]
    for i in range(len(scores["normal_logp"])):
        # Point state is history independent; a fresh fusion avoids carrying episode state.
        r=TwoExpertFusion(config).update(float(scores["normal_logp"][i]),float(scores["abnormal_logp"][i]),int(scores["task_id"][i]))
        out.append(r.state)
    return np.asarray(out)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--visual',type=Path,required=True)
    p.add_argument('--scores',type=Path,required=True); p.add_argument('--fusion',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True); p.add_argument('--report',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=300); p.add_argument('--seed',type=int,default=20260904)
    a=p.parse_args(); v=np.load(a.visual,allow_pickle=False); s=np.load(a.scores,allow_pickle=False)
    if not np.array_equal(v['episode_id'],s['episode_id']): raise ValueError('visual/score row mismatch')
    config=json.loads(a.fusion.read_text(encoding='utf-8')); state=states(s,config)
    clean=np.isclose(v['scale'],1.0); active=v['abnormal'].astype(bool)
    selected=v['valid'].astype(bool)&(state=='ambiguous_overlap')&(clean|active)
    labels=active[selected].astype(np.float32)
    # Visual response + deployable intended action fields + task one-hot; no measured robot response.
    x=np.c_[v['visual'][selected],v['x_condition'][selected,:8],v['x_condition'][selected,58:68]].astype(np.float32)
    mean=x.mean(0); scale=x.std(0); scale[scale<1e-6]=1
    tx=torch.tensor((x-mean)/scale); ty=torch.tensor(labels)
    torch.manual_seed(a.seed); torch.set_num_threads(4)
    model=VisualVerifier(x.shape[1]); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3)
    pos=float(labels.sum()); neg=float(len(labels)-pos); loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg/max(pos,1)))
    for _ in range(a.epochs):
        model.train(); opt.zero_grad(); loss=loss_fn(model(tx),ty); loss.backward(); opt.step()
    model.eval(); payload={'state_dict':model.state_dict(),'input_dim':x.shape[1],'mean':mean,'scale':scale,
        'seed':a.seed,'epochs':a.epochs,'selection':'valid AND ambiguous_overlap AND (clean_normal OR active_fault)',
        'inputs':'agentview visual motion/context + intended action fields + task onehot'}
    torch.save(payload,a.out); digest=hashlib.sha256(a.out.read_bytes()).hexdigest()
    with torch.no_grad(): prob=torch.sigmoid(model(tx)).numpy()
    report={'rows':len(labels),'positive_active_rows':int(labels.sum()),'clean_normal_rows':int((labels==0).sum()),
      'train_loss':float(loss),'positive_probability_median':float(np.median(prob[labels==1])),
      'normal_probability_median':float(np.median(prob[labels==0])),'checkpoint_sha256':digest,
      'warning':'Training fit only; threshold and utility must be decided on seed20 calibration.'}
    a.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
