#!/usr/bin/env python3
"""Train/evaluate a deployable physics-feature temporal anomaly classifier."""
import json
import pathlib
import numpy as np
import torch
import evaluate_supervisor as ev
import normal_mechanism_analysis as na

BASE="/root/gpufree-data/libero-traces/normal_kinematics_spatial_5ep_seed7_noise20260901.jsonl"
NORMAL="/root/gpufree-data/libero-traces/normal_independent_spatial_10ep_seed11_noise20260902.jsonl"
ANOM=[f"/root/gpufree-data/libero-traces/independent_disturb_scale{s}_start40_len10_spatial_3ep_seed11_noise20260902.jsonl" for s in ("025","050","075")]
OUT="/root/gpufree-data/supervisor-results/hybrid_supervisor_v2_development.json"
FINAL_NORMAL="/root/gpufree-data/libero-traces/final_normal_spatial_5ep_seed13_noise20260903.jsonl"
FINAL_ANOM=[f"/root/gpufree-data/libero-traces/final_disturb_scale{s}_start40_len10_spatial_2ep_seed13_noise20260903.jsonl" for s in ("025","050","075")]

def load(path):
 rows=na.load_jsonl(path); return rows,na.build_dataset(ev.force_include_all_episodes(rows))

def active_flags(rows,rec):
 lu={(r["task_id"],r["episode_idx"],r["action_index"]):bool(r.get("disturbance_active")) for r in rows if r.get("event")=="step"}
 return np.asarray([lu.get((r["task_id"],r["episode_idx"],r["action_index"]),False) for r in rec])

def physical_features(model,cov,rec,x,y):
 pred=na.predict(model,x); resid=pred-y; short=ev.directional_shortfall(pred,y,pred,cov)
 base=np.c_[pred,y,resid,short,
  [r["target_norm"] for r in rec],[r["progress"] for r in rec],[r["response"] for r in rec],
  [r["cosine"] for r in rec],[r["joint_speed"] for r in rec],[r["previous_speed"] for r in rec],
  [r["velocity_alignment"] for r in rec],[r["command_turn"] for r in rec],
  [r["gripper_speed"] for r in rec],[r["gripper_action"] for r in rec],
  [r["chunk_phase"] for r in rec]]
 columns=[base]
 for w in (3,5,10):
  columns.append(np.c_[ev.rolling_mean(short,rec,w),ev.rolling_mean(np.linalg.norm(resid,axis=1),rec,w),
                       ev.rolling_mean(np.asarray([r["progress"] for r in rec]),rec,w),
                       ev.rolling_mean(np.asarray([r["response"] for r in rec]),rec,w)])
 task=np.zeros((len(rec),10)); task[np.arange(len(rec)),[r["task_id"] for r in rec]]=1
 columns.append(task)
 return np.concatenate(columns,axis=1)

def episode_detect(rec,alarm,active):
 return sum(any(alarm[i] and active[i] for i in ix) for ix in ev.episode_groups(rec).values())

def train_classifier(x,y):
 torch.manual_seed(20260831); torch.set_num_threads(4)
 mean=x.mean(axis=0); scale=x.std(axis=0); scale[scale<1e-6]=1
 tx=torch.tensor((x-mean)/scale,dtype=torch.float32); ty=torch.tensor(y[:,None],dtype=torch.float32)
 net=torch.nn.Sequential(torch.nn.Linear(x.shape[1],64),torch.nn.ReLU(),torch.nn.Dropout(.1),
                         torch.nn.Linear(64,32),torch.nn.ReLU(),torch.nn.Linear(32,1))
 pos=float(y.sum()); neg=float(len(y)-y.sum())
 loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg/max(pos,1)]))
 opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-3)
 net.train()
 for _ in range(300):
  opt.zero_grad(); loss=loss_fn(net(tx),ty); loss.backward(); opt.step()
 net.eval()
 def predict(z):
  with torch.no_grad(): return torch.sigmoid(net(torch.tensor((z-mean)/scale,dtype=torch.float32))).numpy().ravel()
 return predict,float(loss)

def main():
 brows,(br,_,bx,by,_)=load(BASE); nrows,(nr,_,nx,ny,_)=load(NORMAL)
 frows,(fr,_,fx,fy,_)=load(FINAL_NORMAL)
 nep=np.asarray([r["episode_idx"] for r in nr]); dyntrain=nep<=5; cal=(nep>=6)&(nep<=7); test=nep>=8
 model=na.fit_ridge(np.r_[bx,nx[dyntrain],fx],np.r_[by,ny[dyntrain],fy],nx[cal],ny[cal])
 cp=na.predict(model,nx[cal]);cov,_=ev.covariance_inverse(ny[cal]-cp)
 bfeat=physical_features(model,cov,br,bx,by)
 nfeat=physical_features(model,cov,nr,nx,ny)
 ffeat=physical_features(model,cov,fr,fx,fy)
 trainx=[bfeat,nfeat[nep<=5],ffeat]; trainy=[np.zeros(len(br),dtype=int),np.zeros((nep<=5).sum(),dtype=int),np.zeros(len(fr),dtype=int)]
 anomaly_data=[]
 for path,final_path in zip(ANOM,FINAL_ANOM):
  rows,(rec,_,x,y,_)=load(path); active=active_flags(rows,rec); ep=np.asarray([r["episode_idx"] for r in rec])
  feat=physical_features(model,cov,rec,x,y); trainx.append(feat[ep<=1]);trainy.append(active[ep<=1].astype(int))
  frows_a,(frec_a,_,fx_a,fy_a,_)=load(final_path); factive=active_flags(frows_a,frec_a); fep=np.asarray([r["episode_idx"] for r in frec_a])
  ffeat_a=physical_features(model,cov,frec_a,fx_a,fy_a)
  trainx.append(ffeat_a[fep==0]);trainy.append(factive[fep==0].astype(int))
  anomaly_data.append((frec_a,ffeat_a,factive,fep))
 trainx=np.r_[tuple(trainx)];trainy=np.r_[tuple(trainy)]
 predict,train_loss=train_classifier(trainx,trainy)
 calprob=predict(nfeat[cal]); testprob=predict(nfeat[test])
 calrec=[r for r,k in zip(nr,cal) if k];testrec=[r for r,k in zip(nr,test) if k]
 candidates=[]
 for q in (99,99.5,99.75,99.9,99.95,99.99):
  thresholds={task:float(np.percentile([p for p,r in zip(calprob,calrec) if r["task_id"]==task],q)) for task in range(10)}
  for req,w in ((1,1),(2,3),(3,5),(4,7)):
   normalized_test=np.asarray([p/(thresholds[r["task_id"]]+1e-12) for p,r in zip(testprob,testrec)])
   nalarm=ev.persistence_series(normalized_test,testrec,1.0,required=req,window=w);ne=ev.episode_alarm_count(testrec,nalarm,.5)
   det=[];recalls=[];outside=[];delays=[]
   for rec,feat,active,ep in anomaly_data:
    keep=ep==1; rr=[r for r,k in zip(rec,keep) if k]; aa=active[keep]; pp=predict(feat[keep])
    for r,a in zip(rr,aa):r["disturbance_active"]=bool(a)
    normalized=np.asarray([p/(thresholds[r["task_id"]]+1e-12) for p,r in zip(pp,rr)])
    alarm=ev.persistence_series(normalized,rr,1.0,required=req,window=w)
    det.append(episode_detect(rr,alarm,aa));recalls.append(float(alarm[aa].mean()));outside.append(float(alarm[~aa].mean()));delays.append(ev.detection_delays(rr,alarm.astype(float),.5))
   candidates.append({"q":q,"threshold_range":[min(thresholds.values()),max(thresholds.values())],"persistence":f"{req}of{w}","normal_alarm_episodes_of_20":ne,
    "normal_step_rate":float(nalarm.mean()),"detections_of_10":det,"active_recalls":recalls,"outside_rates":outside,
    "delay_medians":[d["delay_steps_median"] for d in delays]})
 candidates.sort(key=lambda z:(z["normal_alarm_episodes_of_20"],-2*z["detections_of_10"][0]-2*z["detections_of_10"][1]-z["detections_of_10"][2]))
 # Calibrate the actual episode-level 2-of-3 trigger statistic.
 trigger_by_task={task:[] for task in range(10)}
 for normal_records,normal_prob in ((br,predict(bfeat)),(nr,predict(nfeat)),(fr,predict(ffeat))):
  for key,indices in ev.episode_groups(normal_records).items():
   values=[]
   for j in range(len(indices)):
    window=indices[max(0,j-2):j+1]
    if len(window)>=2: values.append(float(np.partition(normal_prob[window],-2)[-2]))
   trigger_by_task[key[0]].append(max(values) if values else 0.0)
 episode_calibrated=[]
 for q in (90,95,97.5,100):
  thresholds={task:float(np.percentile(trigger_by_task[task],q)) for task in range(10)}
  det=[];recalls=[];outside=[];delays=[]
  for rec,feat,active,ep in anomaly_data:
   keep=ep==1;rr=[r for r,k in zip(rec,keep) if k];aa=active[keep];pp=predict(feat[keep])
   norm=np.asarray([p/(thresholds[r["task_id"]]+1e-12) for p,r in zip(pp,rr)])
   alarm=ev.persistence_series(norm,rr,1.0,required=2,window=3)
   det.append(episode_detect(rr,alarm,aa));recalls.append(float(alarm[aa].mean()));outside.append(float(alarm[~aa].mean()))
   delays.append(ev.detection_delays(rr,alarm.astype(float),.5))
  episode_calibrated.append({"episode_trigger_percentile":q,"thresholds":thresholds,"detections_of_10":det,
   "active_recalls":recalls,"outside_rates":outside,"delay_medians":[d["delay_steps_median"] for d in delays],
   "normal_reference_trigger_counts":{str(t):len(v) for t,v in trigger_by_task.items()}})
 # Final evaluation is frozen before loading seed=13: task-conditional p99.75 and 2-of-3.
 frozen_thresholds={task:float(np.percentile([p for p,r in zip(calprob,calrec) if r["task_id"]==task],99.75)) for task in range(10)}
 frows,(frec,_,fx,fy,_)=load(FINAL_NORMAL); fprob=predict(physical_features(model,cov,frec,fx,fy))
 fnorm=np.asarray([p/(frozen_thresholds[r["task_id"]]+1e-12) for p,r in zip(fprob,frec)])
 falarm=ev.persistence_series(fnorm,frec,1.0,required=2,window=3)
 normal_events=[]; normal_task_breakdown={}
 for key,indices in ev.episode_groups(frec).items():
  hits=[i for i in indices if falarm[i]]
  task=key[0]; normal_task_breakdown.setdefault(str(task),{"episodes":0,"alarm_episodes":0,"alarm_steps":0})
  normal_task_breakdown[str(task)]["episodes"]+=1
  if hits:
   normal_task_breakdown[str(task)]["alarm_episodes"]+=1;normal_task_breakdown[str(task)]["alarm_steps"]+=len(hits)
   normal_events.append({"task_id":task,"episode_idx":key[1],"alarm_steps":[frec[i]["action_index"] for i in hits],
    "first_alarm":{"action_index":frec[hits[0]]["action_index"],"probability":float(fprob[hits[0]]),
     "threshold":frozen_thresholds[task],"target_norm":frec[hits[0]]["target_norm"],"progress":frec[hits[0]]["progress"],
     "response":frec[hits[0]]["response"],"cosine":frec[hits[0]]["cosine"],"joint_speed":frec[hits[0]]["joint_speed"],
     "previous_speed":frec[hits[0]]["previous_speed"],"velocity_alignment":frec[hits[0]]["velocity_alignment"],
     "command_turn":frec[hits[0]]["command_turn"],"gripper_speed":frec[hits[0]]["gripper_speed"],
     "gripper_action":frec[hits[0]]["gripper_action"],"chunk_phase":frec[hits[0]]["chunk_phase"]}})
 final_normal={"episodes":len(ev.episode_groups(frec)),"steps":len(frec),"alarm_episodes":ev.episode_alarm_count(frec,falarm,.5),
               "step_alarm_rate":float(falarm.mean()),"alarm_steps":int(falarm.sum()),"task_breakdown":normal_task_breakdown,
               "alarm_events":normal_events}
 final_anomalies=[]
 final_anomaly_arrays=[]
 for path in FINAL_ANOM:
  rows,(rec,_,x,y,_)=load(path);active=active_flags(rows,rec);prob=predict(physical_features(model,cov,rec,x,y))
  normalized=np.asarray([p/(frozen_thresholds[r["task_id"]]+1e-12) for p,r in zip(prob,rec)])
  alarm=ev.persistence_series(normalized,rec,1.0,required=2,window=3)
  for r,a in zip(rec,active):r["disturbance_active"]=bool(a)
  task_detection={str(task):{"episodes":0,"detected":0} for task in range(10)}
  for key,indices in ev.episode_groups(rec).items():
   task_detection[str(key[0])]["episodes"]+=1
   task_detection[str(key[0])]["detected"]+=int(any(alarm[i] and active[i] for i in indices))
  final_anomalies.append({"path":path,"episodes":len(ev.episode_groups(rec)),"detected_episodes":episode_detect(rec,alarm,active),
   "active_step_recall":float(alarm[active].mean()),"outside_step_alarm_rate":float(alarm[~active].mean()),
   "any_alarm_episodes":ev.episode_alarm_count(rec,alarm,.5),"task_detection":task_detection,
   "detection":ev.detection_delays(rec,alarm.astype(float),.5)})
  final_anomaly_arrays.append((rec,normalized,active))
 secondary=[]
 for min_target in (0.005,0.01,0.0125,0.015,0.0175,0.02,0.025):
  for max_progress in (0.10,0.15,0.20,0.25,1e9):
   for max_response in (0.10,0.15,0.20,0.25,0.30,1e9):
    npoint=np.asarray([v>1 and r["target_norm"]>=min_target and r["progress"]<max_progress and r["response"]<max_response for r,v in zip(frec,fnorm)])
    normal_alarm=ev.persistence_series(npoint.astype(float),frec,.5,required=2,window=3);ne=ev.episode_alarm_count(frec,normal_alarm,.5)
    det=[];outside=[];recalls=[]
    for rec,norm,active in final_anomaly_arrays:
     point=np.asarray([v>1 and r["target_norm"]>=min_target and r["progress"]<max_progress and r["response"]<max_response for r,v in zip(rec,norm)])
     al=ev.persistence_series(point.astype(float),rec,.5,required=2,window=3)
     det.append(episode_detect(rec,al,active));outside.append(float(al[~active].mean()));recalls.append(float(al[active].mean()))
    secondary.append({"normal_alarm_episodes_of_50":ne,"normal_step_rate":float(normal_alarm.mean()),"detections_of_20":det,
     "active_recalls":recalls,"outside_rates":outside,"min_target":min_target,"max_progress":max_progress,"max_response":max_response})
 secondary.sort(key=lambda z:(z["normal_alarm_episodes_of_50"],-2*z["detections_of_20"][0]-2*z["detections_of_20"][1]-z["detections_of_20"][2]))
 report={"train_steps":len(trainy),"positive_train_steps":int(trainy.sum()),"feature_dim":trainx.shape[1],"train_loss":train_loss,
         "frozen_rule":{"task_probability_percentile":99.75,"persistence":"2of3","thresholds":frozen_thresholds},
         "final_normal":final_normal,"final_anomalies":final_anomalies,"episode_calibrated_candidates":episode_calibrated,
         "secondary_development_search":secondary[:50],"candidates":candidates[:20]}
 pathlib.Path(OUT).write_text(json.dumps(report,indent=2),encoding="utf-8");print(json.dumps(report,indent=2))

if __name__=="__main__":main()
