#!/usr/bin/env python3
"""Development upper bound: fit all seed31 normal tasks, test paired disturbances."""
from __future__ import annotations
import argparse,collections,json,re
from pathlib import Path
import numpy as np
from evaluate_wrist_response_loso import fit,persistent,score


def base(sample:str)->str:
    m=re.match(r'(.+)_seed31_(?:normal|scale075|scale050)$',sample)
    if not m:raise ValueError(sample)
    return m.group(1)


def main()->None:
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--ridge',type=float,default=10.);p.add_argument('--include-context',action='store_true');a=p.parse_args()
    d=np.load(a.dataset,allow_pickle=False);sid=d['sample_id'].astype(str);valid=d['one_step_valid'].astype(bool);active=d['abnormal'].astype(bool);x=d['x_condition'].astype(float);y=d['one_step_grid'].astype(float)
    if a.include_context:x=np.c_[x,d['wrist_context'].astype(float)]
    clean=valid&np.char.endswith(sid,'_normal');model=fit(x[clean],y[clean],a.ridge);train_scores=score(model,x[clean],y[clean]);thresholds={m:float(np.quantile(v,.99)) for m,v in train_scores.items()};tasks=sorted({base(s) for s in np.unique(sid)});summary={}
    for metric,threshold in thresholds.items():
        normal_fp=normal_alarm=normal_steps=det=active_alarm=active_steps=0;by_scale=collections.defaultdict(lambda:[0,0]);details=[]
        for task in tasks:
            ni=np.flatnonzero(valid&(sid==task+'_seed31_normal'));nv=score(model,x[ni],y[ni])[metric];na=persistent(nv>=threshold);normal_fp+=int(na.any());normal_alarm+=int(na.sum());normal_steps+=len(na)
            td={'task':task,'normal_alarm':bool(na.any()),'scales':{}}
            for tag in ('scale075','scale050'):
                ix=np.flatnonzero(valid&(sid==task+'_seed31_'+tag));values=score(model,x[ix],y[ix])[metric];alarm=persistent(values>=threshold);aa=active[ix];hit=bool(np.any(alarm&aa));det+=int(hit);active_alarm+=int(np.sum(alarm&aa));active_steps+=int(aa.sum());by_scale[tag][0]+=int(hit);by_scale[tag][1]+=1;td['scales'][tag]={'detected':hit,'active_recall':float(np.mean(alarm[aa]))}
            details.append(td)
        summary[metric]={'threshold':threshold,'normal_episode_fp':normal_fp,'normal_episodes':len(tasks),'abnormal_detections':det,'abnormal_episodes':2*len(tasks),'normal_step_alarm_rate':normal_alarm/max(normal_steps,1),'active_step_recall':active_alarm/max(active_steps,1),'by_scale':{k:{'detected':v[0],'episodes':v[1]} for k,v in by_scale.items()},'details':details}
    report={'status':'development paired upper bound; not frozen generalization','normal_rows':int(clean.sum()),'include_context':a.include_context,'input_dim':int(x.shape[1]),'summary':summary};a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='details'} for k,v in summary.items()},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
