#!/usr/bin/env python3
"""Audit dimensionless command-response relations before fitting transfer experts."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from libero_droid_transfer import load_trace


def metrics(command_history_displacement, response):
    result=[]
    rnorm=np.linalg.norm(response,axis=1)
    for lag in range(command_history_displacement.shape[1]):
        command=command_history_displacement[:,lag,:3]; cnorm=np.linalg.norm(command,axis=1)
        dot=np.sum(command*response[:,:3],axis=1); denom=cnorm*cnorm+1e-10
        progress=dot/denom; ratio=rnorm/(cnorm+1e-8); cosine=dot/(cnorm*rnorm+1e-10)
        residual=np.linalg.norm(response[:,:3]-progress[:,None]*command,axis=1)/(rnorm+1e-8)
        result.append(np.stack((progress,ratio,cosine,residual),axis=1))
    return np.stack(result,axis=1)


def auc(labels,scores):
    labels=np.asarray(labels,bool); order=np.argsort(scores); ranks=np.empty(len(scores),float); ranks[order]=np.arange(1,len(scores)+1)
    p=labels.sum(); n=len(labels)-p; return float((ranks[labels].sum()-p*(p+1)/2)/(p*n))


def summary(values):
    return dict(zip(('p01','p05','p50','p95','p99'),map(float,np.percentile(values,[1,5,50,95,99]))))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--droid-data-directory',type=Path,required=True)
    p.add_argument('--libero',type=Path,action='append',required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    manifest=json.loads((a.droid_data_directory/'manifest.json').read_text(encoding='utf-8'))
    droid=np.load(a.droid_data_directory/manifest['output_files']['calibration']['path'],allow_pickle=False)
    droid_metric=metrics(droid['command_history_cartesian_velocity']/15.0,droid['response_eef_delta_xyz_euler'])
    reports={'droid_normal':{},'libero':[],'lag_auc_low_progress':{}}
    for lag in range(6): reports['droid_normal'][str(lag)]={name:summary(droid_metric[:,lag,i]) for i,name in enumerate(('progress','ratio','cosine','orthogonal_fraction'))}
    all_m=[]; all_y=[]
    for path in a.libero:
        data,meta=load_trace(path); m=metrics(data['command_history_cartesian_velocity']/20.0,data['response_eef_delta_xyz_euler'])
        all_m.append(m); all_y.append(meta['abnormal'])
        reports['libero'].append({'path':str(path),'rows':len(m),'abnormal_rows':int(meta['abnormal'].sum()),
            'by_lag':{str(lag):{'normal_progress':summary(m[~meta['abnormal'],lag,0]),
                                'abnormal_progress':summary(m[meta['abnormal'],lag,0]) if meta['abnormal'].any() else None}
                      for lag in range(6)}})
    joined=np.concatenate(all_m); labels=np.concatenate(all_y)
    for lag in range(6): reports['lag_auc_low_progress'][str(lag)]=auc(labels,-joined[:,lag,0])
    a.output.write_text(json.dumps(reports,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'lag_auc_low_progress':reports['lag_auc_low_progress']},indent=2))
if __name__=='__main__':main()
