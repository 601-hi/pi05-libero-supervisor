#!/usr/bin/env python3
"""Robust task-agnostic hand-eye proxy: proprioceptive pose -> image ROI center."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
def phi(x):
 pos=x[:,22:25];quat=x[:,25:29];quad=np.c_[pos[:,0]**2,pos[:,1]**2,pos[:,2]**2,pos[:,0]*pos[:,1],pos[:,0]*pos[:,2],pos[:,1]*pos[:,2]]
 return np.c_[np.ones(len(x)),pos,quat,quad]
def predict(model,x):return phi(x)@model['weights']
def main():
 p=argparse.ArgumentParser();p.add_argument('--train',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--model-out',type=Path,required=True);p.add_argument('--report-out',type=Path,required=True);p.add_argument('--ridge',type=float,default=1e-3);a=p.parse_args();tr=np.load(a.train,allow_pickle=False);ca=np.load(a.calibration,allow_pickle=False);sel=tr['valid'].astype(bool)&np.isclose(tr['scale'],1);z=phi(tr['x_condition'][sel]);target=tr['center_before'][sel].astype(float);keep=np.ones(len(z),bool)
 for _ in range(4):
  w=np.linalg.solve(z[keep].T@z[keep]+a.ridge*np.eye(z.shape[1]),z[keep].T@target[keep]);error=np.linalg.norm(target-z@w,axis=1);keep=error<=np.quantile(error,.9)
 model={'weights':w,'ridge':np.asarray(a.ridge)};np.savez_compressed(a.model_out,**model);valid=ca['valid'].astype(bool)&np.isclose(ca['scale'],1);prediction=predict(model,ca['x_condition'][valid]);error=np.linalg.norm(ca['center_before'][valid]-prediction,axis=1);inside=np.all((prediction>=0)&(prediction<224),axis=1);report={'method':'robust quadratic proprioceptive pose to distal image center','task_identity_used':False,'train_rows_initial':int(sel.sum()),'train_rows_after_trim':int(keep.sum()),'calibration_clean_rows':int(valid.sum()),'calibration_pixel_error':{'median':float(np.median(error)),'p75':float(np.quantile(error,.75)),'p90':float(np.quantile(error,.9)),'p95':float(np.quantile(error,.95))},'prediction_inside_image_fraction':float(inside.mean())}
 a.report_out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
