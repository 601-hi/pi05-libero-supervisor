#!/usr/bin/env python3
"""Leave-one-task-out evaluation on the seed31 wrist development collection."""
from __future__ import annotations
import argparse, collections, json, re
from pathlib import Path
import numpy as np
from evaluate_wrist_response_loso import auc, fit, persistent, score, support_mask


def task_key(sample: str) -> str:
    match=re.match(r'(.+)_seed31_(?:normal|scale075|scale050)$',sample)
    if not match:raise ValueError(sample)
    return match.group(1)


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--ridge',type=float,default=10.0);p.add_argument('--include-context',action='store_true');a=p.parse_args()
    d=np.load(a.dataset,allow_pickle=False);sid=d['sample_id'].astype(str);keys=np.asarray([task_key(x) for x in sid]);valid=d['one_step_valid'].astype(bool);active=d['abnormal'].astype(bool)
    x=d['x_condition'].astype(float)
    if a.include_context:x=np.c_[x,d['wrist_context'].astype(float)]
    y=d['one_step_grid'].astype(float);normal_sample=np.char.endswith(sid,'_normal');tasks=sorted(np.unique(keys));folds=[]
    totals=collections.defaultdict(lambda:{'normal_episode_fp':0,'abnormal_episode_detections':0,'abnormal_episodes':0,'active_alarm':0,'active_steps':0,'normal_alarm':0,'normal_steps':0})
    by_scale=collections.defaultdict(lambda:collections.defaultdict(lambda:[0,0]))
    for held in tasks:
        train=valid&normal_sample&(keys!=held);model=fit(x[train],y[train],a.ridge);train_scores=score(model,x[train],y[train]);thresholds={m:float(np.quantile(v,.99)) for m,v in train_scores.items()}
        held_normal=np.flatnonzero(valid&normal_sample&(keys==held));test_active=np.flatnonzero(valid&active&(keys==held));test=held_normal.tolist()+test_active.tolist();labels=np.r_[np.zeros(len(held_normal),bool),np.ones(len(test_active),bool)]
        test_scores=score(model,x[test],y[test]);fold={'held_task':held,'train_normal_rows':int(train.sum()),'normal_rows':len(held_normal),'active_rows':len(test_active),'auc':{m:auc(labels,v) for m,v in test_scores.items()},'metrics':{}}
        for metric,threshold in thresholds.items():
            normal_values=score(model,x[held_normal],y[held_normal])[metric];normal_points=normal_values>=threshold;normal_alarm=persistent(normal_points)
            entry={'threshold':threshold,'normal_step_alarm_rate':float(normal_points.mean()),'normal_episode_alarm':bool(normal_alarm.any()),'scales':{}}
            total=totals[metric];total['normal_episode_fp']+=int(normal_alarm.any());total['normal_alarm']+=int(normal_alarm.sum());total['normal_steps']+=len(normal_alarm)
            for tag,scale in [('scale075',.75),('scale050',.5)]:
                ix=np.flatnonzero(valid&(sid==held+'_seed31_'+tag));values=score(model,x[ix],y[ix])[metric];alarm=persistent(values>=threshold);aa=active[ix]
                detected=bool(np.any(alarm&aa));entry['scales'][tag]={'active_point_recall':float(np.mean(values[aa]>=threshold)),'active_persistent_recall':float(np.mean(alarm[aa])),'detected':detected}
                total['abnormal_episodes']+=1;total['abnormal_episode_detections']+=int(detected);total['active_alarm']+=int(np.sum(alarm&aa));total['active_steps']+=int(aa.sum());by_scale[metric][tag][0]+=int(detected);by_scale[metric][tag][1]+=1
            fold['metrics'][metric]=entry
        supported,_=support_mask(x[train],x[held_normal]);active_supported,_=support_mask(x[train],x[test_active]);fold['support']={'normal_coverage':float(supported.mean()),'active_coverage':float(active_supported.mean())}
        folds.append(fold)
    summary={}
    for metric,total in totals.items():
        summary[metric]={**total,'active_step_recall':total['active_alarm']/max(total['active_steps'],1),'normal_step_alarm_rate':total['normal_alarm']/max(total['normal_steps'],1),
          'by_scale':{tag:{'detected':v[0],'episodes':v[1]} for tag,v in by_scale[metric].items()}}
    report={'status':'seed31 development leave-one-task-out; thresholds refit per fold','tasks':tasks,'normal_fit_only':True,'include_context':a.include_context,'input_dim':int(x.shape[1]),'summary':summary,'folds':folds}
    a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({'tasks':len(tasks),'summary':summary},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
