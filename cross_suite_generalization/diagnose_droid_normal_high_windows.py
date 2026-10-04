#!/usr/bin/env python3
"""Robot-state audit of high-LR normal DROID windows (source-only)."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

LOW=np.asarray([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-.0175,-2.8973])
HIGH=np.asarray([2.8973,1.7628,2.8973,-.0698,2.8973,3.7525,2.8973])

def summary(x):
 x=np.asarray(x,float);return {'mean':float(x.mean()),'p10':float(np.percentile(x,10)),'median':float(np.median(x)),'p90':float(np.percentile(x,90))}

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-directory',type=Path,required=True);p.add_argument('--scores',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 manifest=json.loads((a.data_directory/'manifest.json').read_text(encoding='utf-8'));cal_path=Path(manifest['output_files']['calibration']['path']);cal_path=cal_path if cal_path.is_absolute() else a.data_directory/cal_path
 data=np.load(cal_path);score=np.load(a.scores);counts=[int(r['exported_transitions']) for r in manifest['episodes'] if r['split']=='calibration' and r['accepted']]
 episode=np.concatenate([np.full(n,i,np.int32) for i,n in enumerate(counts)]);tune=episode%2==1
 lr=(score['abnormal_logp']-score['normal_logp'])[~score['abnormal']]
 if len(lr)!=tune.sum():raise ValueError('score/data alignment failed')
 q=data['state_joint_position'][tune];eef=data['state_eef_pose_xyz_euler'][tune];grip=data['state_gripper_position'][tune,0];cmd=data['command_history_cartesian_velocity'][tune,0,:3]/15.;resp=data['response_eef_delta_xyz_euler'][tune,:3]
 qmargin=np.min(np.minimum((q-LOW)/(HIGH-LOW),(HIGH-q)/(HIGH-LOW)),axis=1);cnorm=np.linalg.norm(cmd,axis=1);rnorm=np.linalg.norm(resp,axis=1);progress=np.sum(cmd*resp,axis=1)/(cnorm**2+1e-10)
 tune_episode=episode[tune];high=np.zeros(len(lr),bool);events=[]
 for ep in np.unique(tune_episode):
  idx=np.flatnonzero(tune_episode==ep);rolling=np.convolve(lr[idx],np.ones(10)/10,mode='valid');start=int(np.argmax(rolling));chosen=idx[start:start+10];high[chosen]=True
  events.append({'episode':int(ep),'start':start,'mean_lr':float(rolling[start]),'gripper_median':float(np.median(grip[chosen])),'joint_margin_median':float(np.median(qmargin[chosen])),'eef_xyz_median':np.median(eef[chosen,:3],axis=0).tolist(),'command_norm_median':float(np.median(cnorm[chosen])),'progress_median':float(np.median(progress[chosen]))})
 features={'gripper_position':grip,'joint_limit_margin':qmargin,'eef_x':eef[:,0],'eef_y':eef[:,1],'eef_z':eef[:,2],'command_norm':cnorm,'response_norm':rnorm,'directed_progress':progress}
 result={'source_only':True,'normal_steps':len(lr),'max_window_steps':int(high.sum()),'comparison':{name:{'all_tune_normal':summary(values),'episode_max_lr_windows':summary(values[high])} for name,values in features.items()},'events':events,
 'limits':['joint-limit margin is a deployable kinematic proxy, not a Jacobian singular-value proof','DROID gripper units must not be assumed identical to LIBERO units']}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
