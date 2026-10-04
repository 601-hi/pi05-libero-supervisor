#!/usr/bin/env python3
"""Normal-only task-agnostic predictor for distal ROI directional flow."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def auc(y,v):
 p=v[y];n=v[~y];return float(((p[:,None]>n[None,:]).sum()+.5*(p[:,None]==n[None,:]).sum())/(len(p)*len(n)))
def fit(d,ridge):
 valid=d['valid'].astype(bool);clean=np.isclose(d['scale'],1);sel=valid&clean;x=np.c_[d['x_condition'][:,:-10],d['center_before']/224.].astype(np.float64);y=d['local_flow'].astype(np.float64);xm=x[sel].mean(0);xs=x[sel].std(0);xs[xs<1e-6]=1;ym=y[sel].mean(0);xn=(x[sel]-xm)/xs;w=np.linalg.solve(xn.T@xn+ridge*np.eye(xn.shape[1]),xn.T@(y[sel]-ym));return {'x_mean':xm,'x_scale':xs,'y_mean':ym,'weights':w,'ridge':np.asarray(ridge)}
def score(m,d):
 x=np.c_[d['x_condition'][:,:-10],d['center_before']/224.].astype(np.float64);y=d['local_flow'].astype(np.float64);p=m['y_mean']+((x-m['x_mean'])/m['x_scale'])@m['weights'];yn=np.linalg.norm(y,axis=1);pn=np.linalg.norm(p,axis=1);return {'residual':np.linalg.norm(y-p,axis=1)/(pn+1e-6),'directed_response':np.sum(y*p,axis=1)/(pn*pn+1e-6),'cosine':np.sum(y*p,axis=1)/(yn*pn+1e-6),'predicted_norm':pn,'actual_norm':yn}
def main():
 p=argparse.ArgumentParser();p.add_argument('--train',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--model-out',type=Path,required=True);p.add_argument('--scores-out',type=Path,required=True);p.add_argument('--report-out',type=Path,required=True);p.add_argument('--ridge',type=float,default=10.);a=p.parse_args();tr=np.load(a.train,allow_pickle=False);ca=np.load(a.calibration,allow_pickle=False);m=fit(tr,a.ridge);s=score(m,ca);np.savez_compressed(a.model_out,**m);np.savez_compressed(a.scores_out,**s,valid=ca['valid'],abnormal=ca['abnormal'],episode_id=ca['episode_id'],action_index=ca['action_index'],scale=ca['scale']);v=ca['valid'].astype(bool);y=ca['abnormal'].astype(bool);report={'method':'distal ROI 2x2 directional flow response','appearance_used':False,'task_identity_used':False,'anomaly_labels_used_for_fit':False,'train_valid_clean':int((tr['valid']&np.isclose(tr['scale'],1)).sum()),'calibration_valid':int(v.sum()),'auc':{'high_residual':auc(y[v],s['residual'][v]),'low_response':auc(y[v],-s['directed_response'][v]),'low_cosine':auc(y[v],-s['cosine'][v])}}
 a.report_out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
