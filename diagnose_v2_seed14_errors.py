#!/usr/bin/env python3
"""Mechanistic audit of frozen-v2 false positives and false negatives."""
import json
import pathlib
import numpy as np
import evaluate_supervisor as ev
import train_hybrid_supervisor_v2 as v2
import evaluate_v2_seed14_frozen as frozen

OUT="/root/gpufree-data/supervisor-results/v2_seed14_error_diagnosis.json"
NAMES=([f"pred_{a}" for a in "xyz"]+[f"actual_{a}" for a in "xyz"]+[f"residual_{a}" for a in "xyz"]+
 ["shortfall","target_norm","progress","response","cosine","joint_speed","previous_speed","velocity_alignment",
  "command_turn","gripper_speed","gripper_action","chunk_phase"]+
 [f"rolling_{metric}_w{w}" for w in (3,5,10) for metric in ("shortfall","residual_norm","progress","response")]+
 [f"task_{i}" for i in range(10)])

def episode_rows(path,model,cov,predict,thresholds,scale,anomaly):
    rows,(rec,_,x,y,_)=v2.load(path); feat=v2.physical_features(model,cov,rec,x,y);prob=predict(feat)
    norm=np.asarray([p/(thresholds[r["task_id"]]+1e-12) for p,r in zip(prob,rec)])
    alarm=ev.persistence_series(norm,rec,1.0,required=2,window=3)
    active=v2.active_flags(rows,rec) if anomaly else np.zeros(len(rec),bool)
    result=[]
    for key,indices in ev.episode_groups(rec).items():
        if anomaly:
            focus=[i for i in indices if active[i]]
            detected=any(alarm[i] and active[i] for i in indices)
            label="TP" if detected else "FN"
        else:
            # Use a causal 10-step window ending at the largest 2-of-3 trigger statistic.
            best_j=1;best=-np.inf
            for j in range(1,len(indices)):
                window=indices[max(0,j-2):j+1];value=float(np.partition(prob[window],-2)[-2])
                if value>best:best=value;best_j=j
            focus=indices[max(0,best_j-9):best_j+1]
            detected=any(alarm[i] for i in indices);label="FP" if detected else "TN"
        vector=np.r_[feat[focus].mean(axis=0),feat[focus].std(axis=0)]
        result.append({"category":label,"scale":scale,"task_id":key[0],"episode_idx":key[1],
          "focus_actions":[rec[i]["action_index"] for i in focus],"max_probability":float(max(prob[i] for i in focus)),
          "max_normalized_risk":float(max(norm[i] for i in focus)),"feature_mean":feat[focus].mean(axis=0).tolist(),
          "feature_std":feat[focus].std(axis=0).tolist(),"vector":vector.tolist()})
    return result

def effect(records,a,b):
    xa=np.asarray([r["feature_mean"] for r in records if r["category"]==a]);xb=np.asarray([r["feature_mean"] for r in records if r["category"]==b])
    pooled=np.sqrt((xa.var(axis=0)+xb.var(axis=0))/2+1e-12);d=(xa.mean(axis=0)-xb.mean(axis=0))/pooled
    order=np.argsort(-np.abs(d))
    return [{"feature":NAMES[i],"cohen_d":float(d[i]),"mean_a":float(xa[:,i].mean()),"mean_b":float(xb[:,i].mean())} for i in order[:15]]

def nearest_overlap(records,source,target):
    allx=np.asarray([r["vector"] for r in records]);mean=allx.mean(axis=0);std=allx.std(axis=0);std[std<1e-8]=1
    s=[r for r in records if r["category"]==source];t=[r for r in records if r["category"]==target]
    rows=[]
    for r in s:
        x=(np.asarray(r["vector"])-mean)/std;best=None
        for q in t:
            if q["task_id"]!=r["task_id"]:continue
            dist=float(np.mean((x-(np.asarray(q["vector"])-mean)/std)**2))
            if best is None or dist<best[0]:best=(dist,q)
        rows.append({"source":{"task":r["task_id"],"episode":r["episode_idx"],"scale":r["scale"]},
          "nearest_same_task":None if best is None else {"distance":best[0],"episode":best[1]["episode_idx"],"scale":best[1]["scale"]}})
    return rows

def main():
    model,cov,predict,thresholds,loss=frozen.fit_frozen();records=[]
    records+=episode_rows(frozen.FINAL_NORMAL,model,cov,predict,thresholds,"normal",False)
    for path,scale in zip(frozen.FINAL_ANOM,("0.25","0.50","0.75")):
        records+=episode_rows(path,model,cov,predict,thresholds,scale,True)
    counts={c:sum(r["category"]==c for r in records) for c in ("TN","FP","TP","FN")}
    task_counts={c:{str(t):sum(r["category"]==c and r["task_id"]==t for r in records) for t in range(10)} for c in counts}
    compact=[{k:r[k] for k in ("category","scale","task_id","episode_idx","focus_actions","max_probability","max_normalized_risk")} for r in records if r["category"] in ("FP","FN")]
    report={"counts":counts,"task_counts":task_counts,"errors":compact,"effects":{"FP_vs_TP":effect(records,"FP","TP"),"FN_vs_TN":effect(records,"FN","TN")},
      "nearest":{"FP_to_TP":nearest_overlap(records,"FP","TP"),"FN_to_TN":nearest_overlap(records,"FN","TN")}}
    pathlib.Path(OUT).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
