#!/usr/bin/env python3
"""Causal offline replay of the source-calibrated execution candidate."""
from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from vla_supervisor.conditional_execution import ConditionalDualExpertScorer

def main():
 p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,action='append',required=True);p.add_argument('--model',type=Path,action='append',required=True);p.add_argument('--threshold',type=float,required=True);p.add_argument('--reliability-calibration',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 scorer=ConditionalDualExpertScorer(a.model,reliability_calibration=a.reliability_calibration); episodes=[]
 for path in a.trace:
  rows=[json.loads(x) for x in path.open(encoding='utf-8') if x.strip()]; groups=defaultdict(list); ends={}
  for r in rows:
   key=(int(r.get('task_id',-1)),int(r.get('episode_idx',0)))
   if r.get('event')=='step':groups[key].append(r)
   elif r.get('event')=='episode_end':ends[key]=r
  for key,steps in groups.items():
   scorer.reset();history=[];alarms=[];active=[];scores=[];unknown=[]
   for r in sorted(steps,key=lambda z:int(z['action_index'])):
    b={'eef_pos':r['eef_pos_before'],'joint_pos':r['joint_pos_before'],'gripper_qpos':r['gripper_qpos_before']};q={'eef_pos':r['eef_pos_after'],'joint_pos':r['joint_pos_after'],'gripper_qpos':r['gripper_qpos_after']}
    s,c,e=scorer(r['intended_action'],b,q,history);alarm=bool(c>=.5 and s>a.threshold);alarms.append(alarm);unknown.append(e.get('support_status')=='unknown_low_support');active.append(bool(r.get('disturbance_active',False)));scores.append(s if np.isfinite(s) else None);history.append({'intended_action':r['intended_action']})
   rules={}
   for name,k,m in [('1of1',1,1),('2of3',2,3),('3of5',3,5),('4of7',4,7)]:
    hits=[]
    for i in range(len(alarms)):
     hit=sum(alarms[max(0,i-m+1):i+1])>=k;hits.append(hit)
    first=next((i for i,v in enumerate(hits) if v),None)
    rules[name]={'any_alarm':any(hits),'first_alarm_action':first,
                 'active_detection':any(v and active[i] for i,v in enumerate(hits))}
   end=ends.get(key,{});episodes.append({'path':str(path),'task_id':key[0],'episode_idx':key[1],'success':end.get('success'),'steps':len(steps),'alarms':sum(alarms),'any_alarm':any(alarms),'unknown_low_support_steps':sum(unknown),'active_steps':sum(active),'active_detections':sum(x and y for x,y in zip(alarms,active)),'inactive_alarms':sum(x and not y for x,y in zip(alarms,active)),'max_sequence_score':max(x for x in scores if x is not None),'temporal_rules':rules})
 rule_summary={name:{'failed_detected':sum(x['success'] is False and x['temporal_rules'][name]['any_alarm'] for x in episodes),'successful_false_alarms':sum(x['success'] is True and x['temporal_rules'][name]['any_alarm'] for x in episodes)} for name in ('1of1','2of3','3of5','4of7')}
 result={'threshold':a.threshold,'threshold_origin':'DROID source-only frozen sequence calibration','target_data_used_for_threshold':False,'episodes':episodes,'summary':{'episodes':len(episodes),'alarmed_episodes':sum(x['any_alarm'] for x in episodes),'successful_episodes':sum(x['success'] is True for x in episodes),'failed_episodes':sum(x['success'] is False for x in episodes),'failed_episodes_alarmed':sum(x['success'] is False and x['any_alarm'] for x in episodes),'successful_episodes_alarmed':sum(x['success'] is True and x['any_alarm'] for x in episodes),'temporal_rules':rule_summary}}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result['summary'],indent=2))
if __name__=='__main__':main()
