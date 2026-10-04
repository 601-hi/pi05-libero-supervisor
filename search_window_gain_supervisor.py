#!/usr/bin/env python3
"""Search a causal rolling directed-gain supervisor."""
import numpy as np
import evaluate_supervisor as ev
import normal_mechanism_analysis as na

BASE="/root/gpufree-data/libero-traces/normal_kinematics_spatial_5ep_seed7_noise20260901.jsonl"
NORMAL="/root/gpufree-data/libero-traces/normal_independent_spatial_10ep_seed11_noise20260902.jsonl"
ANOM=[f"/root/gpufree-data/libero-traces/independent_disturb_scale{s}_start40_len10_spatial_3ep_seed11_noise20260902.jsonl" for s in ("025","050","075")]

def load(path):
 rows=na.load_jsonl(path); return rows,na.build_dataset(ev.force_include_all_episodes(rows))

def rolling_gain(pred,actual,records,window):
 out=np.zeros(len(records)); energy=np.zeros(len(records))
 for ix in ev.episode_groups(records).values():
  for j,i in enumerate(ix):
   use=ix[max(0,j-window+1):j+1]; p=pred[use]; a=actual[use]
   den=float(np.sum(p*p)); out[i]=float(np.sum(p*a)/(den+1e-12)); energy[i]=den**.5
 return out,energy

def anomaly_data(model,path,window):
 rows,(rec,_,x,y,_)=load(path); pred=na.predict(model,x); gain,energy=rolling_gain(pred,y,rec,window)
 lu={(r["task_id"],r["episode_idx"],r["action_index"]):bool(r.get("disturbance_active")) for r in rows if r.get("event")=="step"}
 active=np.asarray([lu.get((r["task_id"],r["episode_idx"],r["action_index"]),False) for r in rec])
 for r,a in zip(rec,active):r["disturbance_active"]=bool(a)
 return rec,gain,energy,active

def detected(rec,alarm,active):
 return sum(any(alarm[i] and active[i] for i in ix) for ix in ev.episode_groups(rec).values())

def main():
 _,(br,_,bx,by,_)=load(BASE); _,(nr,_,nx,ny,_)=load(NORMAL)
 ep=np.asarray([r["episode_idx"] for r in nr]); dev=ep<=5; cal=(ep>=6)&(ep<=7); test=ep>=8
 model=na.fit_ridge(np.r_[bx,nx[dev]],np.r_[by,ny[dev]],nx[cal],ny[cal])
 refx=np.r_[bx,nx[dev],nx[cal]]; refy=np.r_[by,ny[dev],ny[cal]]; refr=br+[r for r,k in zip(nr,dev|cal) if k]
 testpred=na.predict(model,nx[test]); testr=[r for r,k in zip(nr,test) if k]; testy=ny[test]
 results=[]
 for w in (2,3,4,5,7,10,12,15):
  rg,re=rolling_gain(na.predict(model,refx),refy,refr,w); tg,te=rolling_gain(testpred,testy,testr,w)
  aa=[anomaly_data(model,p,w) for p in ANOM]
  for q in (.1,.25,.5,1,2,2.5,5,7.5,10):
   th={t:float(np.percentile([g for g,r in zip(rg,refr) if r["task_id"]==t],q)) for t in range(10)}
   for emin in (.001,.002,.003,.005,.0075,.01,.015,.02):
    alarm=np.asarray([g<th[r["task_id"]] and e>=emin for r,g,e in zip(testr,tg,te)])
    ne=ev.episode_alarm_count(testr,alarm,.5)
    if ne>2:continue
    det=[];recall=[];outside=[];delays=[]
    for rec,gain,energy,active in aa:
     al=np.asarray([g<th[r["task_id"]] and e>=emin for r,g,e in zip(rec,gain,energy)])
     det.append(detected(rec,al,active));recall.append(float(al[active].mean()));outside.append(float(al[~active].mean()))
     delays.append(ev.detection_delays(rec,al.astype(float),.5))
    utility=2*det[0]+2*det[1]+det[2]-10*ne
    results.append((utility,ne,det,recall,outside,delays,w,q,emin,th))
 results.sort(key=lambda z:(-z[0],z[1],-sum(z[2])))
 print("NORMAL_TEST_EPISODES",len(ev.episode_groups(testr)))
 for z in results[:40]:
  u,ne,d,rec,out,delay,w,q,emin,th=z
  print({"utility":u,"normal_alarm_episodes":ne,"detections_of_30":d,"active_recalls":rec,"outside_rates":out,
   "delay_medians":[x["delay_steps_median"] for x in delay],"window":w,"lower_percentile":q,"min_predicted_energy":emin,
   "threshold_range":[min(th.values()),max(th.values())]})

if __name__=="__main__":main()
