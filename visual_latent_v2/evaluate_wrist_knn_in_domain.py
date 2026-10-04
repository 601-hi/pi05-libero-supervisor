#!/usr/bin/env python3
"""Local normal-response expert with episode-level calibration on seed31 development."""
from __future__ import annotations
import argparse,collections,json,re
from pathlib import Path
import numpy as np
from evaluate_wrist_response_loso import persistent


def base(sample:str)->str:
    m=re.match(r'(.+)_seed31_(?:normal|scale075|scale050)$',sample)
    if not m:raise ValueError(sample)
    return m.group(1)


def predict_knn(train_x,train_y,query_x,k,exclude_self=False):
    if exclude_self and len(query_x)!=len(train_x):raise ValueError('self exclusion requires matching rows')
    predictions=[];nearest=[];train_norm=np.sum(train_x*train_x,axis=1)
    for start in range(0,len(query_x),128):
        query=query_x[start:start+128]
        squared=np.maximum(np.sum(query*query,axis=1)[:,None]+train_norm[None,:]-2*query@train_x.T,0)
        distance=np.sqrt(squared)
        if exclude_self:
            rows=np.arange(len(query));distance[rows,start+rows]=np.inf
        neighbors=np.argpartition(distance,k-1,axis=1)[:,:k];near=np.take_along_axis(distance,neighbors,axis=1)
        weight=1/(near+1e-3);weight/=weight.sum(1,keepdims=True)
        predictions.append((train_y[neighbors]*weight[...,None]).sum(1));nearest.append(near[:,0])
    return np.concatenate(predictions),np.concatenate(nearest)


def scores(actual,predicted,normal_residual_center,normal_residual_scale):
    an=np.linalg.norm(actual,axis=1);pn=np.linalg.norm(predicted,axis=1);res=np.linalg.norm(actual-predicted,axis=1)
    return {'high_residual':(res-normal_residual_center)/normal_residual_scale,
      'low_response':-np.sum(actual*predicted,axis=1)/(pn*pn+1e-12),
      'low_cosine':-np.sum(actual*predicted,axis=1)/(an*pn+1e-12)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--k',type=int,default=8);a=p.parse_args()
    d=np.load(a.dataset,allow_pickle=False);sid=d['sample_id'].astype(str);valid=d['one_step_valid'].astype(bool);active=d['abnormal'].astype(bool);x=d['x_condition'].astype(float);y=d['one_step_grid'].astype(float)
    clean=valid&np.char.endswith(sid,'_normal');xm=x[clean].mean(0);xs=x[clean].std(0);xs[xs<1e-8]=1;tx=(x[clean]-xm)/xs;ty=y[clean]
    loo,loo_distance=predict_knn(tx,ty,tx,a.k,True);normal_residual=np.linalg.norm(ty-loo,axis=1);rc=float(np.median(normal_residual));rs=float(1.4826*np.median(np.abs(normal_residual-rc)));rs=max(rs,float(normal_residual.std()),1e-6)
    prediction,distance=predict_knn(tx,ty,(x-xm)/xs,a.k);all_scores=scores(y,prediction,rc,rs)
    loo_scores=scores(ty,loo,rc,rs)
    for metric in all_scores:all_scores[metric][clean]=loo_scores[metric]
    tasks=sorted({base(s) for s in np.unique(sid)});normal_distance_floor=float(np.quantile(loo_distance,.99))
    reports={}
    for metric,value in all_scores.items():
        normal_sequences=[]
        for task in tasks:
            ix=np.flatnonzero(valid&(sid==task+'_seed31_normal'));normal_sequences.append(value[ix])
        candidates=np.r_[np.unique(np.concatenate(normal_sequences)),np.inf]
        budgets={}
        for budget in (0,1,2):
            choices=[]
            for threshold in candidates:
                fp=sum(bool(persistent(seq>=threshold).any()) for seq in normal_sequences);det=recall=0
                if fp>budget:continue
                for task in tasks:
                    for tag in ('scale075','scale050'):
                        ix=np.flatnonzero(valid&(sid==task+'_seed31_'+tag));alarm=persistent(value[ix]>=threshold);aa=active[ix];det+=int(np.any(alarm&aa));recall+=int(np.sum(alarm&aa))
                choices.append((det,recall,-fp,float(threshold)))
            det,recall,negfp,threshold=max(choices);by_scale={}
            for tag in ('scale075','scale050'):
                hits=0
                for task in tasks:
                    ix=np.flatnonzero(valid&(sid==task+'_seed31_'+tag));hits+=int(np.any(persistent(value[ix]>=threshold)&active[ix]))
                by_scale[tag]={'detected':hits,'episodes':len(tasks)}
            budgets[str(budget)]={'threshold':threshold,'normal_episode_fp':-negfp,'abnormal_detections':det,'abnormal_episodes':2*len(tasks),'active_step_recall':recall/(20*len(tasks)),'by_scale':by_scale}
        reports[metric]={'budgets':budgets}
    report={'status':'paired seed31 development calibration only','k':a.k,'normal_rows':int(clean.sum()),'normal_support_nn_q99':normal_distance_floor,'reports':reports}
    a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
