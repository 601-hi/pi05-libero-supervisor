#!/usr/bin/env python3
"""Export causal distal-arm ROI flow aligned with an existing visual dataset."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import cv2,numpy as np
from visualize_distal_motion_roi import distal_roi,optical,read

def local_feature(flow,center,radius=26):
 h,w=flow.shape[:2];cx,cy=center;yy,xx=np.mgrid[:h,:w];dx=xx-cx;dy=yy-cy;r2=dx*dx+dy*dy
 ring=(r2>36**2)&(r2<58**2);background=np.median(flow[ring],axis=0) if ring.any() else np.median(flow.reshape(-1,2),axis=0)
 features=[]
 for sy in (-1,1):
  for sx in (-1,1):
   cell=(r2<=radius**2)&(sx*dx>=0)&(sy*dy>=0)
   value=np.median(flow[cell],axis=0) if cell.any() else np.zeros(2)
   features.extend((value-background).tolist())
 return np.asarray(features,np.float32),np.asarray(background,np.float32)
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();d=np.load(a.source,allow_pickle=False);meta=json.loads(str(d['metadata_json']));paths=meta['videos'];n=len(d['episode_id']);local=np.zeros((n,8),np.float32);center_before=np.zeros((n,2),np.float32);center_after=np.zeros((n,2),np.float32);background=np.zeros((n,2),np.float32);valid=np.zeros(n,bool)
 for eid,path in paths.items():
  ix=np.flatnonzero(d['episode_id']==eid);ix=ix[np.argsort(d['action_index'][ix])];frames=read(path)
  if len(frames)!=len(ix):raise ValueError(f'{eid}: {len(frames)} frames != {len(ix)} rows')
  previous_center=None
  for j,i in enumerate(ix):
   if j+1>=len(frames):continue
   f=optical(frames[j],frames[j+1]);before=None if previous_center is None else np.asarray(previous_center).copy();current,_,_=distal_roi(f,previous_center)
   if current is not None and before is not None:
    local[i],background[i]=local_feature(f,current);center_before[i]=before;center_after[i]=current;valid[i]=True
   previous_center=current
 a.out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.out,local_flow=local,background_flow=background,center_before=center_before,center_after=center_after,valid=valid,x_condition=d['x_condition'],abnormal=d['abnormal'],task_id=d['task_id'],episode_id=d['episode_id'],action_index=d['action_index'],scale=d['scale'],metadata_json=np.asarray(json.dumps({'schema':1,'roi':'distal border-connected motion component, radius26, 2x2 quadrants','causal_input':'center_before only','appearance_used':False,'source':str(a.source)},ensure_ascii=False)));print({'rows':n,'valid':int(valid.sum()),'episodes':len(paths)})
if __name__=='__main__':main()
