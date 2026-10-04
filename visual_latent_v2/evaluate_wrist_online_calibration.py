#!/usr/bin/env python3
"""Episode-local warm-up calibration for wrist response (development experiment)."""
from __future__ import annotations
import argparse,collections,json,re
from pathlib import Path
import numpy as np
from evaluate_wrist_response_loso import persistent


def task(sample):
    m=re.match(r'(.+)_seed31_(?:normal|scale075|scale050)$',sample)
    if not m:raise ValueError(sample)
    return m.group(1)


def episode_scores(x,y,warmup,ridge):
    fit=np.arange(min(warmup,len(x)));xm=x[fit].mean(0);xs=x[fit].std(0);xs[xs<1e-8]=1;ym=y[fit].mean(0);xn=(x[fit]-xm)/xs
    weights=np.linalg.solve(xn.T@xn+ridge*np.eye(xn.shape[1]),xn.T@(y[fit]-ym));prediction=ym+((x-xm)/xs)@weights
    an=np.linalg.norm(y,axis=1);pn=np.linalg.norm(prediction,axis=1);res=np.linalg.norm(y-prediction,axis=1);fit_res=res[fit];center=np.median(fit_res);scale=max(1.4826*np.median(np.abs(fit_res-center)),fit_res.std(),1e-6)
    return {'high_residual':(res-center)/scale,'low_response':-np.sum(y*prediction,axis=1)/(pn*pn+1e-12),'low_cosine':-np.sum(y*prediction,axis=1)/(an*pn+1e-12)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--warmup',type=int,default=20);p.add_argument('--ridge',type=float,default=10.);a=p.parse_args()
    d=np.load(a.dataset,allow_pickle=False);sid=d['sample_id'].astype(str);valid=d['one_step_valid'].astype(bool);active=d['abnormal'].astype(bool);full_x=d['x_condition'].astype(float);y=d['one_step_grid'].astype(float)
    # Compact observable condition: current and previous intended action, joint velocity, gripper state/velocity.
    x=full_x[:,np.r_[0:14,28:39]];samples=sorted(np.unique(sid));scores_by_sample={}
    for sample in samples:
        ix=np.flatnonzero(valid&(sid==sample));scores_by_sample[sample]=(ix,episode_scores(x[ix],y[ix],a.warmup,a.ridge))
    normal_samples=[s for s in samples if s.endswith('_normal')];tasks=sorted({task(s) for s in samples});reports={}
    for metric in ('high_residual','low_response','low_cosine'):
        candidates=np.r_[np.unique(np.concatenate([scores_by_sample[s][1][metric][a.warmup:] for s in normal_samples])),np.inf];budgets={}
        for budget in (0,1,2):
            choices=[]
            for threshold in candidates:
                fp=sum(bool(persistent(scores_by_sample[s][1][metric]>=threshold)[a.warmup:].any()) for s in normal_samples)
                if fp>budget:continue
                det=recall=0
                for t in tasks:
                    for tag in ('scale075','scale050'):
                        sample=t+'_seed31_'+tag;ix,sv=scores_by_sample[sample];alarm=persistent(sv[metric]>=threshold);aa=active[ix];det+=int(np.any(alarm&aa));recall+=int(np.sum(alarm&aa))
                choices.append((det,recall,-fp,float(threshold)))
            det,recall,negfp,threshold=max(choices);by_scale={}
            for tag in ('scale075','scale050'):
                hits=0
                for t in tasks:
                    ix,sv=scores_by_sample[t+'_seed31_'+tag];hits+=int(np.any(persistent(sv[metric]>=threshold)&active[ix]))
                by_scale[tag]={'detected':hits,'episodes':len(tasks)}
            budgets[str(budget)]={'threshold':threshold,'normal_episode_fp':-negfp,'abnormal_detections':det,'abnormal_episodes':2*len(tasks),'active_step_recall':recall/(20*len(tasks)),'by_scale':by_scale}
        reports[metric]={'budgets':budgets}
    report={'status':'seed31 development; episode-local startup calibration','warmup_steps':a.warmup,'assumption':'first warmup_steps are nominal','condition_dim':int(x.shape[1]),'reports':reports}
    a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
