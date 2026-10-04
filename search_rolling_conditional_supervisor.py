#!/usr/bin/env python3
"""Search finite-memory, task-conditional shortfall detectors."""

import numpy as np
import evaluate_supervisor as ev
import normal_mechanism_analysis as na

BASE="/root/gpufree-data/libero-traces/normal_kinematics_spatial_5ep_seed7_noise20260901.jsonl"
NORMAL="/root/gpufree-data/libero-traces/normal_independent_spatial_10ep_seed11_noise20260902.jsonl"
ANOM=[f"/root/gpufree-data/libero-traces/independent_disturb_scale{s}_start40_len10_spatial_3ep_seed11_noise20260902.jsonl" for s in ("025","050","075")]

def load(path):
 rows=na.load_jsonl(path); return rows,na.build_dataset(ev.force_include_all_episodes(rows))

def scored(model,cov,path):
 rows,(rec,_,x,y,_)=load(path); pred=na.predict(model,x)
 score=ev.directional_shortfall(pred,y,pred,cov)
 lookup={(r["task_id"],r["episode_idx"],r["action_index"]):bool(r.get("disturbance_active")) for r in rows if r.get("event")=="step"}
 active=np.asarray([lookup.get((r["task_id"],r["episode_idx"],r["action_index"]),False) for r in rec])
 for r,a in zip(rec,active): r["disturbance_active"]=bool(a)
 return rec,score,active

def detect_count(rec,alarm,active):
 return sum(any(alarm[i] and active[i] for i in ix) for ix in ev.episode_groups(rec).values())

def main():
 _,(br,_,bx,by,_)=load(BASE); _,(nr,_,nx,ny,_)=load(NORMAL)
 ep=np.asarray([r["episode_idx"] for r in nr]); dev=ep<=5; cal=(ep>=6)&(ep<=7); test=ep>=8
 model=na.fit_ridge(np.r_[bx,nx[dev]],np.r_[by,ny[dev]],nx[cal],ny[cal])
 cp=na.predict(model,nx[cal]); cov,_=ev.covariance_inverse(ny[cal]-cp)
 # Threshold reference deliberately spans many normal episodes per task. The
 # held-out episode 8--9 records remain untouched for internal false alarms.
 ref_x=np.r_[bx,nx[dev],nx[cal]]; ref_y=np.r_[by,ny[dev],ny[cal]]
 cr=br+[r for r,k in zip(nr,dev|cal) if k]
 ref_pred=na.predict(model,ref_x)
 cs=ev.directional_shortfall(ref_pred,ref_y,ref_pred,cov)
 tp=na.predict(model,nx[test]); ts=ev.directional_shortfall(tp,ny[test],tp,cov); tr=[r for r,k in zip(nr,test) if k]
 aa=[scored(model,cov,p) for p in ANOM]; results=[]
 for window in (2,3,4,5,7,10):
  croll=ev.rolling_mean(cs,cr,window); troll=ev.rolling_mean(ts,tr,window)
  aroll=[(r,ev.rolling_mean(s,r,window),a) for r,s,a in aa]
  for q in (95,97,97.5,98,98.5,99,99.25,99.5,99.75,99.9):
   th={t:float(np.percentile([v for v,r in zip(croll,cr) if r["task_id"]==t],q)) for t in range(10)}
   for mt in (0.0,0.005,0.01,0.015,0.02,0.025):
    alarm=np.asarray([v>th[r["task_id"]] and r["target_norm"]>=mt for r,v in zip(tr,troll)])
    ne=ev.episode_alarm_count(tr,alarm,.5)
    if ne>2: continue
    det=[]; recalls=[]; outside=[]; delays=[]
    for rec,roll,active in aroll:
     al=np.asarray([v>th[r["task_id"]] and r["target_norm"]>=mt for r,v in zip(rec,roll)])
     det.append(detect_count(rec,al,active)); recalls.append(float(al[active].mean())); outside.append(float(al[~active].mean()))
     delays.append(ev.detection_delays(rec,al.astype(float),.5))
    utility=2*det[0]+2*det[1]+det[2]-10*ne
    results.append((utility,ne,det,recalls,outside,delays,window,q,mt,th))
 results.sort(key=lambda x:(-x[0],x[1],-sum(x[2])))
 print("NORMAL_TEST_EPISODES",len(ev.episode_groups(tr)))
 for x in results[:40]:
  utility,ne,det,recalls,outside,delays,w,q,mt,th=x
  print({"utility":utility,"normal_alarm_episodes":ne,"detections_of_30":det,"active_recalls":recalls,
         "outside_rates":outside,"delay_medians":[d["delay_steps_median"] for d in delays],"window":w,"q":q,
         "min_target":mt,"threshold_range":[min(th.values()),max(th.values())]})

if __name__=="__main__": main()
