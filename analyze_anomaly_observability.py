#!/usr/bin/env python3
"""Quantify whether fixed-time translation faults are actually excited."""
import json
import numpy as np

ROOT="/root/gpufree-data/libero-traces"
FILES=[f"{ROOT}/independent_disturb_scale{s}_start40_len10_spatial_3ep_seed11_noise20260902.jsonl" for s in ("025","050","075")]

for path in FILES:
 rows=[json.loads(x) for x in open(path,encoding="utf-8") if x.strip()]
 steps=[r for r in rows if r.get("event")=="step" and r.get("disturbance_active")]
 groups={}
 for r in steps: groups.setdefault((r["task_id"],r["episode_idx"]),[]).append(r)
 vals=[]
 for key,rr in groups.items():
  intended=np.asarray([r["intended_target_translation"] for r in rr],float)
  actual=np.asarray([np.asarray(r["eef_pos_after"])-np.asarray(r["eef_pos_before"]) for r in rr])
  command_path=float(np.linalg.norm(intended,axis=1).sum())
  vector_command=float(np.linalg.norm(intended.sum(axis=0)))
  actual_path=float(np.linalg.norm(actual,axis=1).sum())
  vals.append((key,command_path,vector_command,actual_path,actual_path/(command_path+1e-12)))
 vals.sort(key=lambda x:x[1])
 arr=np.asarray([v[1:] for v in vals])
 print("FILE",path)
 print("COMMAND_PATH_M",{f"p{q}":float(np.percentile(arr[:,0],q)) for q in (0,10,25,50,75,90,100)})
 print("VECTOR_COMMAND_M",{f"p{q}":float(np.percentile(arr[:,1],q)) for q in (0,25,50,75,100)})
 print("ACTUAL_PATH_M",{f"p{q}":float(np.percentile(arr[:,2],q)) for q in (0,25,50,75,100)})
 print("LOW_EXCITATION_COUNTS",{str(t):int(np.sum(arr[:,0]<t)) for t in (.05,.10,.20,.30)})
 print("LOWEST",vals[:8])
