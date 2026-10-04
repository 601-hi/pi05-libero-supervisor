#!/usr/bin/env python3
"""Robot-kinematic description of reviewed false-alarm windows."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
def features(r):
 a=np.asarray(r['intended_action'],float);cmd=.05*np.clip(a[:3],-1,1);actual=np.asarray(r['eef_pos_after'])-np.asarray(r['eef_pos_before']);cn=np.linalg.norm(cmd);an=np.linalg.norm(actual)
 return {'command_translation_m':cn,'actual_translation_m':an,'directed_progress':float(cmd@actual/(cn*cn+1e-12)),'direction_cosine':float(cmd@actual/(cn*an+1e-12)),'rotation_action_norm':float(np.linalg.norm(a[3:6])),'joint_velocity_norm':float(np.linalg.norm(r['joint_vel_before'])),'gripper_opening_m':float(abs(r['gripper_qpos_before'][0]-r['gripper_qpos_before'][1]))}
def summary(rows):return {k:{'median':float(np.median([x[k] for x in rows])),'p10':float(np.percentile([x[k] for x in rows],10)),'p90':float(np.percentile([x[k] for x in rows],90))} for k in rows[0]}
def main():
 p=argparse.ArgumentParser();p.add_argument('--queue',type=Path,required=True);p.add_argument('--trace',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();q=json.loads(a.queue.read_text(encoding='utf-8'))
 steps={};ends={}
 for line in a.trace.open(encoding='utf-8'):
  r=json.loads(line);ep=int(r.get('episode_idx',0))
  if r.get('event')=='step':steps.setdefault(ep,[]).append(r)
  elif r.get('event')=='episode_end':ends[ep]=r
 selected=[];indices=[]
 for item in q['items']:
  if item['suite']=='libero_spatial' and item['task_id']==9 and item['reason']=='execution_false_alarm_candidate' and item['episode_idx']>=5:
   lo,hi=item['review_action_range'];center=(lo+hi)//2;indices.append({'episode_idx':item['episode_idx'],'center':center})
   selected.extend(features(r) for r in steps[item['episode_idx']] if abs(int(r['action_index'])-center)<=2)
 baseline=[features(r) for ep,rs in steps.items() if ends.get(ep,{}).get('success') for r in rs]
 result={'interpretation':'reviewed false alarms are visually normal approach/grasp-contact windows','reviewed_windows':indices,'reviewed_step_count':len(selected),'successful_task_step_count':len(baseline),'reviewed':summary(selected),'all_successful_task_steps':summary(baseline),'median_percentile_within_successful_task':{k:float(100*np.mean([x[k] for x in baseline]<=np.median([y[k] for y in selected]))) for k in selected[0]}}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
