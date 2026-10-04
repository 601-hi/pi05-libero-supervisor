#!/usr/bin/env python3
"""Visual diagnostic for paired normal/disturbed local motion (not a detector)."""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2,numpy as np

def frames(path):
 c=cv2.VideoCapture(str(path));out=[]
 while True:
  ok,f=c.read()
  if not ok:break
  out.append(f)
 c.release();return out
def flow(prev,curr):
 a=cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY);b=cv2.cvtColor(curr,cv2.COLOR_BGR2GRAY)
 return cv2.calcOpticalFlowFarneback(a,b,None,.5,3,15,3,5,1.2,0)
def heat(value):
 scale=np.percentile(value,99);u=np.uint8(np.clip(value/(scale+1e-6)*255,0,255));return cv2.applyColorMap(u,cv2.COLORMAP_TURBO)
def boxes(frame,magnitude):
 threshold=np.percentile(magnitude,92);mask=np.uint8(magnitude>=threshold)*255;mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
 n,_,stats,_=cv2.connectedComponentsWithStats(mask)
 out=frame.copy()
 for x,y,w,h,area in stats[1:]:
  if area>=30:cv2.rectangle(out,(x,y),(x+w,y+h),(0,255,255),2)
 return out
def label(image,text):
 out=image.copy();cv2.rectangle(out,(0,0),(out.shape[1],22),(0,0,0),-1);cv2.putText(out,text,(4,16),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1,cv2.LINE_AA);return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--normal',type=Path,required=True);p.add_argument('--disturbed',type=Path,required=True);p.add_argument('--indices',default='80,81,82,84,86,90');p.add_argument('--out',type=Path,required=True);a=p.parse_args();nf=frames(a.normal);df=frames(a.disturbed);rows=[]
 for i in map(int,a.indices.split(',')):
  fn=nf[i];fd=df[i];mn=np.linalg.norm(flow(nf[max(i-1,0)],fn),axis=2);md=np.linalg.norm(flow(df[max(i-1,0)],fd),axis=2);pair=cv2.absdiff(fn,fd).mean(2)
  tiles=[label(fn,f'normal frame {i}'),label(fd,f'scale075 frame {i}'),label(heat(mn),'normal flow'),label(heat(md),'scale075 flow'),label(heat(pair),'paired difference'),label(boxes(fd,md),'motion ROI candidates')]
  rows.append(np.concatenate(tiles,axis=1))
 canvas=np.concatenate(rows,axis=0);a.out.parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(a.out),canvas);print({'out':str(a.out),'shape':canvas.shape,'normal_frames':len(nf),'disturbed_frames':len(df)})
if __name__=='__main__':main()
