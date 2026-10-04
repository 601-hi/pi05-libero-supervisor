#!/usr/bin/env python3
"""Strict frozen v2 evaluation on untouched seed=14 traces."""
import json
import pathlib
import numpy as np
import evaluate_supervisor as ev
import normal_mechanism_analysis as na
import train_hybrid_supervisor_v2 as v2

FINAL_NORMAL="/root/gpufree-data/libero-traces/v2_final_normal_spatial_5ep_seed14_noise20260904.jsonl"
FINAL_ANOM=[f"/root/gpufree-data/libero-traces/v2_final_disturb_scale{s}_start40_len10_spatial_2ep_seed14_noise20260904.jsonl" for s in ("025","050","075")]
OUT="/root/gpufree-data/supervisor-results/hybrid_supervisor_v2_frozen_seed14.json"

def trigger_values(records,probabilities):
    result={}
    for key,indices in ev.episode_groups(records).items():
        values=[]
        for j in range(len(indices)):
            window=indices[max(0,j-2):j+1]
            if len(window)>=2:
                values.append(float(np.partition(probabilities[window],-2)[-2]))
        result[key]=max(values) if values else 0.0
    return result

def fit_frozen():
    _,(br,_,bx,by,_)=v2.load(v2.BASE)
    _,(nr,_,nx,ny,_)=v2.load(v2.NORMAL)
    _,(fr,_,fx,fy,_)=v2.load(v2.FINAL_NORMAL)
    nep=np.asarray([r["episode_idx"] for r in nr]); dyntrain=nep<=5; cal=(nep>=6)&(nep<=7)
    model=na.fit_ridge(np.r_[bx,nx[dyntrain],fx],np.r_[by,ny[dyntrain],fy],nx[cal],ny[cal])
    cp=na.predict(model,nx[cal]);cov,_=ev.covariance_inverse(ny[cal]-cp)
    bfeat=v2.physical_features(model,cov,br,bx,by)
    nfeat=v2.physical_features(model,cov,nr,nx,ny)
    ffeat=v2.physical_features(model,cov,fr,fx,fy)
    trainx=[bfeat,nfeat[nep<=5],ffeat]
    trainy=[np.zeros(len(br),int),np.zeros(int((nep<=5).sum()),int),np.zeros(len(fr),int)]
    for path,final_path in zip(v2.ANOM,v2.FINAL_ANOM):
        rows,(rec,_,x,y,_)=v2.load(path); active=v2.active_flags(rows,rec); ep=np.asarray([r["episode_idx"] for r in rec])
        feat=v2.physical_features(model,cov,rec,x,y);trainx.append(feat[ep<=1]);trainy.append(active[ep<=1].astype(int))
        rows,(rec,_,x,y,_)=v2.load(final_path);active=v2.active_flags(rows,rec);ep=np.asarray([r["episode_idx"] for r in rec])
        feat=v2.physical_features(model,cov,rec,x,y);trainx.append(feat[ep==0]);trainy.append(active[ep==0].astype(int))
    predict,loss=v2.train_classifier(np.r_[tuple(trainx)],np.r_[tuple(trainy)])
    trigger_by_task={t:[] for t in range(10)}
    for rec,feat in ((br,bfeat),(nr,nfeat),(fr,ffeat)):
        for key,value in trigger_values(rec,predict(feat)).items():trigger_by_task[key[0]].append(value)
    thresholds={t:float(np.percentile(trigger_by_task[t],95)) for t in range(10)}
    return model,cov,predict,thresholds,loss

def evaluate(path,model,cov,predict,thresholds,anomaly):
    rows,(rec,_,x,y,_)=v2.load(path);feat=v2.physical_features(model,cov,rec,x,y);prob=predict(feat)
    normalized=np.asarray([p/(thresholds[r["task_id"]]+1e-12) for p,r in zip(prob,rec)])
    alarm=ev.persistence_series(normalized,rec,1.0,required=2,window=3)
    active=v2.active_flags(rows,rec) if anomaly else np.zeros(len(rec),bool)
    for r,a in zip(rec,active):r["disturbance_active"]=bool(a)
    episodes=[]
    for key,indices in ev.episode_groups(rec).items():
        hits=[i for i in indices if alarm[i]]; active_indices=[i for i in indices if active[i]]
        detected=any(alarm[i] and active[i] for i in indices)
        focus=active_indices if active_indices else indices
        episodes.append({"task_id":key[0],"episode_idx":key[1],"alarm":bool(hits),"detected":bool(detected),
          "alarm_actions":[rec[i]["action_index"] for i in hits],"max_probability":float(max(prob[i] for i in focus)),
          "max_normalized_risk":float(max(normalized[i] for i in focus)),
          "mean_target_norm":float(np.mean([rec[i]["target_norm"] for i in focus])),
          "mean_progress":float(np.mean([rec[i]["progress"] for i in focus])),
          "mean_response":float(np.mean([rec[i]["response"] for i in focus])),
          "mean_cosine":float(np.mean([rec[i]["cosine"] for i in focus]))})
    report={"path":path,"episodes":len(episodes),"steps":len(rec),"alarm_episodes":sum(e["alarm"] for e in episodes),
      "step_alarm_rate":float(alarm.mean()),"episodes_detail":episodes}
    if anomaly:
        report.update({"detected_episodes":sum(e["detected"] for e in episodes),"active_steps":int(active.sum()),
          "active_step_recall":float(alarm[active].mean()),"outside_step_alarm_rate":float(alarm[~active].mean()),
          "detection":ev.detection_delays(rec,alarm.astype(float),.5)})
    return report

def main():
    model,cov,predict,thresholds,loss=fit_frozen()
    report={"frozen_rule":{"episode_trigger_percentile":95,"persistence":"2of3","thresholds":thresholds,
      "training_seeds":[7,11,13],"evaluation_seed":14,"train_loss":loss},
      "normal":evaluate(FINAL_NORMAL,model,cov,predict,thresholds,False),
      "anomalies":[evaluate(p,model,cov,predict,thresholds,True) for p in FINAL_ANOM]}
    pathlib.Path(OUT).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"frozen_rule":report["frozen_rule"],"normal":{k:v for k,v in report["normal"].items() if k!="episodes_detail"},
      "anomalies":[{k:v for k,v in a.items() if k!="episodes_detail"} for a in report["anomalies"]]},ensure_ascii=False,indent=2))

if __name__=="__main__":main()
