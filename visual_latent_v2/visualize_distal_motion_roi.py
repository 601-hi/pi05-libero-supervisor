#!/usr/bin/env python3
"""Visualize a geometry-free distal moving-link ROI for a border-mounted arm."""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2,numpy as np

def read(path):
 c=cv2.VideoCapture(str(path));frames=[]
 while True:
  ok,f=c.read()
  if not ok:break
  frames.append(f)
 c.release();return frames
def optical(prev,curr):
 a=cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY);b=cv2.cvtColor(curr,cv2.COLOR_BGR2GRAY)
 return cv2.calcOpticalFlowFarneback(a,b,None,.5,3,15,3,5,1.2,0)
def distal_roi(flow,previous_center=None):
 mag=np.linalg.norm(flow,axis=2);positive=mag[mag>.02]
 if not len(positive):return previous_center,None,None
 threshold=max(.08,float(np.percentile(positive,75)));mask=np.uint8(mag>=threshold)
 mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8));n,labels,stats,_=cv2.connectedComponentsWithStats(mask)
 candidates=[];h,w=mask.shape
 for label in range(1,n):
  x,y,bw,bh,area=stats[label]
  touches=x<=3 or y<=3
  if touches and area>=40:candidates.append((area,label))
 if not candidates:return previous_center,mask,None
 label=max(candidates)[1];ys,xs=np.where(labels==label);border=(xs<=5)|(ys<=5)
 if not border.any():return previous_center,mask,None
 # Prefer the top mounting boundary.  The moving arm can later touch the left
 # image edge; treating that contact as the base pulls the ROI back to the
 # wrist housing instead of the gripper.
 top=ys<=5;left=xs<=5
 mounting=top if top.any() else left
 anchor=np.array([np.median(xs[mounting]),np.median(ys[mounting])]);distance=np.sqrt((xs-anchor[0])**2+(ys-anchor[1])**2)
 distal=distance>=np.percentile(distance,94);weights=mag[ys[distal],xs[distal]]+1e-6
 center=np.array([np.average(xs[distal],weights=weights),np.average(ys[distal],weights=weights)])
 if previous_center is not None:center=.65*np.asarray(previous_center)+.35*center
 return center,mask,anchor
def main():
 p=argparse.ArgumentParser();p.add_argument('--video',type=Path,required=True);p.add_argument('--indices',default='70,75,80,81,82,84,86,90,100');p.add_argument('--out',type=Path,required=True);a=p.parse_args();frames=read(a.video);wanted=set(map(int,a.indices.split(',')));tiles=[];center=None
 for i in range(1,len(frames)):
  f=optical(frames[i-1],frames[i]);center,mask,anchor=distal_roi(f,center)
  if i not in wanted:continue
  image=frames[i].copy()
  if center is not None:
   c=tuple(np.round(center).astype(int));cv2.circle(image,c,24,(0,0,255),2);cv2.circle(image,c,4,(0,0,255),-1)
  if anchor is not None:cv2.circle(image,tuple(np.round(anchor).astype(int)),5,(255,0,0),-1)
  cv2.putText(image,f'frame {i}: red distal ROI, blue base',(4,16),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1,cv2.LINE_AA)
  tiles.append(image)
 rows=[np.concatenate(tiles[i:i+3],axis=1) for i in range(0,len(tiles),3)];canvas=np.concatenate(rows,axis=0);a.out.parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(a.out),canvas);print({'out':str(a.out),'shape':canvas.shape})
if __name__=='__main__':main()
