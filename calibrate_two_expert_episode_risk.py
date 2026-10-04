#!/usr/bin/env python3
"""Calibrate expert competition against complete-episode false-alarm risk."""
from __future__ import annotations

import argparse, collections, json
from pathlib import Path
import numpy as np


EPS=1e-9
def robust(v):
    med=float(np.median(v)); scale=float(np.subtract(*np.percentile(v,[75,25]))/1.349)
    return med,max(scale,1e-3)

def persistent(values, threshold):
    point=np.asarray(values)>=threshold; out=np.zeros(len(point),bool)
    for i in range(len(point)):
        out[i]=point[max(0,i-2):i+1].sum()>=2
    return out

def rolling_second(values):
    out=np.full(len(values),-np.inf)
    for i in range(1,len(values)):
        w=np.asarray(values[max(0,i-2):i+1]); out[i]=np.partition(w,-2)[-2]
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument("--scores",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True); p.add_argument("--max-normal-episode-fp",type=int,default=1)
    p.add_argument("--global-calibration",action="store_true",
                   help="Use one task-ID-free calibration distribution and emit no by_task table")
    a=p.parse_args(); d=np.load(a.scores,allow_pickle=False)
    nlp,alp=d["normal_logp"],d["abnormal_logp"]; ratio=alp-nlp
    task=d["task_id"].astype(int); active=d["abnormal"].astype(bool); clean=np.isclose(d["scale"],1.0)
    by_task={}; rz=np.zeros(len(ratio)); nz=np.zeros(len(ratio)); abnormal_floor=np.zeros(len(ratio));global_fields={}
    groups_to_fit=[(None,np.ones(len(task),bool))] if a.global_calibration else [(int(t),task==t) for t in sorted(np.unique(task))]
    for t,scope in groups_to_fit:
        m=scope&clean; rc,rs=robust(ratio[m]); nc,ns=robust(nlp[m]);am=scope&active
        fields={"ratio_center":rc,"ratio_scale":rs,"normal_logp_center":nc,"normal_logp_scale":ns,
                "normal_absolute_floor":float(np.quantile(nlp[m],.005)),
                "abnormal_absolute_floor":float(np.quantile(alp[am],.01))}
        if t is None: global_fields=fields
        else: by_task[str(t)]=fields
        rz[scope]=(ratio[scope]-rc)/rs;nz[scope]=(nc-nlp[scope])/ns;abnormal_floor[scope]=fields["abnormal_absolute_floor"]
    groups=collections.defaultdict(list)
    for i,e in enumerate(d["episode_id"]): groups[str(e)].append(i)
    episode_cache=[]; normal_triggers=[]
    for key,ix in groups.items():
        ix=sorted(ix,key=lambda i:int(d["action_index"][i]))
        evidence=np.where(alp[ix]>=abnormal_floor[ix],rz[ix],-np.inf)
        trigger=rolling_second(evidence)
        episode_cache.append((ix,trigger))
        if clean[ix].all(): normal_triggers.append(float(np.max(trigger)))
    candidates=np.unique(np.r_[normal_triggers,[np.nextafter(v,np.inf) for v in normal_triggers],-np.inf,np.inf])
    choices=[]
    for th in candidates:
        normal_fp=0; detections=0; recalled=0
        for ix,trigger in episode_cache:
            alarm=trigger>=th
            if clean[ix].all(): normal_fp+=bool(alarm.any())
            elif active[ix].any(): detections+=bool(np.any(alarm & active[ix])); recalled+=int(np.sum(alarm & active[ix]))
        if normal_fp<=a.max_normal_episode_fp: choices.append((detections,recalled,-normal_fp,float(th)))
    if not choices: raise RuntimeError("no ratio threshold satisfies normal episode risk budget")
    detections,recalled,negfp,threshold=max(choices)
    # Unknown novelty gets its own episode-level normal budget and is not tuned on known anomalies.
    normal_trigger=[]
    for ix in groups.values():
        if clean[ix].all():
            ix=sorted(ix,key=lambda i:int(d["action_index"][i])); vals=nz[ix]
            windows=[np.partition(vals[max(0,j-2):j+1],-2)[-2] for j in range(1,len(vals))]
            normal_trigger.append(max(windows) if windows else float("inf"))
    unknown_threshold=float(np.quantile(normal_trigger,.90))
    config={"version":2,"calibration":"episode_risk_budget_global" if a.global_calibration else "episode_risk_budget","persistence_required":2,"persistence_window":3,
            "ratio_abnormal_threshold":threshold,"unknown_novelty_threshold":unknown_threshold,
            **global_fields, **({} if a.global_calibration else {"by_task":by_task}),
            "calibration_summary":{"normal_episode_budget":a.max_normal_episode_fp,
            "known_normal_fp":-negfp,"known_abnormal_episode_detections":detections,
            "known_active_alarm_steps":recalled,"normal_episodes":len(normal_trigger)}}
    a.out.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
    printable = dict(config["calibration_summary"])
    printable.update({"ratio_threshold_z": threshold, "unknown_threshold_z": unknown_threshold})
    print(json.dumps(printable, indent=2))

if __name__=="__main__": main()
