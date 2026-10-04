#!/usr/bin/env python3
"""Comprehensive sealed LIBERO-10 audit with strict and normal-only tracks."""
from __future__ import annotations
import argparse,json,re
from pathlib import Path
import numpy as np
from evaluate_transfer_scores import causal_mean

PATTERN=re.compile(r'libero_10_task(?P<task>\d+)_seed32_(?P<condition>normal|scale025|scale050|scale075)')

def metrics(score,label,file_id,task,episode,action,threshold,selected=None):
 alarm=score>threshold; selected=np.ones(len(score),bool) if selected is None else selected
 active=selected&label;inactive=selected&~label; episode_hits=[];delays=[];normal_false=[]
 for key in sorted(set(zip(file_id[selected],task[selected],episode[selected]))):
  mask=selected&np.asarray([x==key for x in zip(file_id,task,episode)])
  if (mask&label).any():
   onset=action[mask&label].min(); hits=action[mask&label&alarm]
   episode_hits.append(bool(len(hits))); 
   if len(hits):delays.append(int(hits.min()-onset))
  elif mask.any():normal_false.append(bool((mask&alarm).any()))
 return {'threshold':float(threshold),'active_step_recall':float(alarm[active].mean()) if active.any() else None,
         'inactive_step_alarm_rate':float(alarm[inactive].mean()) if inactive.any() else None,
         'abnormal_episode_detection':float(np.mean(episode_hits)) if episode_hits else None,'abnormal_episodes':len(episode_hits),
         'normal_episode_false_alarm':float(np.mean(normal_false)) if normal_false else None,'normal_episodes':len(normal_false),
         'first_alarm_delay_steps_median':float(np.median(delays)) if delays else None}

def main():
 p=argparse.ArgumentParser();p.add_argument('--scores',type=Path,required=True);p.add_argument('--trace-root',type=Path,required=True)
 p.add_argument('--source-q95',type=float,required=True);p.add_argument('--source-q99',type=float,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 d=np.load(a.scores);smooth=causal_mean(d['score'],d['file_index'],d['task_id'],d['episode_idx'],d['action_index']);label=d['abnormal'].astype(bool)
 files=sorted(a.trace_root.glob('*.jsonl')); condition_by_file={};task_by_file={}
 for i,path in enumerate(files):
  m=PATTERN.fullmatch(path.stem)
  if not m:continue
  condition_by_file[i]=m['condition'];task_by_file[i]=int(m['task'])
 condition=np.asarray([condition_by_file[int(i)] for i in d['file_index']]);
 adapt=(condition=='normal')&(d['episode_idx']==0); normal_test=(condition=='normal')&(d['episode_idx']==1)
 target_q95=float(np.percentile(smooth[adapt],95));target_q99=float(np.percentile(smooth[adapt],99))
 evaluate_mask=(condition!='normal')|normal_test
 result={'strict_zero_shot':{'source_q95':metrics(smooth,label,d['file_index'],d['task_id'],d['episode_idx'],d['action_index'],a.source_q95),
                             'source_q99_primary':metrics(smooth,label,d['file_index'],d['task_id'],d['episode_idx'],d['action_index'],a.source_q99)},
         'exploratory_normal_only_adaptation':{'fit_normal_episodes':5,'test_normal_episodes':5,'target_q95':metrics(smooth,label,d['file_index'],d['task_id'],d['episode_idx'],d['action_index'],target_q95,evaluate_mask),
                                                'target_q99':metrics(smooth,label,d['file_index'],d['task_id'],d['episode_idx'],d['action_index'],target_q99,evaluate_mask)},
         'by_condition_at_source_q99':{},'policy_success':{}}
 for name in ('scale025','scale050','scale075'):
  mask=condition==name;result['by_condition_at_source_q99'][name]=metrics(smooth,label,d['file_index'],d['task_id'],d['episode_idx'],d['action_index'],a.source_q99,mask)
 for name in ('normal','scale025','scale050','scale075'):
  ends=[]
  for path in files:
   m=PATTERN.fullmatch(path.stem)
   if m and m['condition']==name:
    for line in path.open(encoding='utf-8'):
     row=json.loads(line)
     if row.get('event')=='episode_end':ends.append(bool(row['success']))
  result['policy_success'][name]={'episodes':len(ends),'successes':sum(ends),'rate':float(np.mean(ends))}
 a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
