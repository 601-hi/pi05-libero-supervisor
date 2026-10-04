#!/usr/bin/env python3
"""Cross-tab weak mechanistic evidence without relabeling every failure as mismatch."""
from __future__ import annotations
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path

def key(d): return (str(d.get('suite','')),int(d['task_id']),int(d['episode_idx']))
def suite_from_path(path):
 n=Path(path).name
 return 'libero_spatial' if 'libero_spatial' in n else 'libero_90' if 'libero_90' in n else 'unknown'
def main():
 p=argparse.ArgumentParser();p.add_argument('--execution',type=Path,required=True);p.add_argument('--stall',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 ex=json.loads(a.execution.read_text(encoding='utf-8'));signals={}
 for path in a.stall:
  report=json.loads(path.read_text(encoding='utf-8'))
  for block in report['results']:
   if block['signal'] not in ('low_progress_run','gripper_reversals_window'):continue
   for d in block['details']:signals.setdefault(key(d),{})[block['signal']]=bool(d['alarm'])
 rows=[]
 for e in ex['episodes']:
  k=(suite_from_path(e['path']),int(e['task_id']),int(e['episode_idx']));s=signals.get(k,{})
  execution=bool(e['temporal_rules']['1of1']['any_alarm']);low=bool(s.get('low_progress_run',False));grip=bool(s.get('gripper_reversals_window',False))
  if grip:mechanism='policy_gripper_oscillation'
  elif low:mechanism='persistent_low_progress_ambiguous_contact_or_stall'
  elif e['success'] is False:mechanism='outcome_or_policy_failure_without_motion_alarm'
  else:mechanism='no_weak_failure_evidence'
  rows.append({'suite':k[0],'task_id':k[1],'episode_idx':k[2],'success':e['success'],'execution_candidate_alarm':execution,'low_progress_alarm':low,'gripper_oscillation_alarm':grip,'weak_mechanism':mechanism})
 def summarize(items):
  return {'episodes':len(items),'successes':sum(r['success'] is True for r in items),'failures':sum(r['success'] is False for r in items),'execution_alarms':sum(r['execution_candidate_alarm'] for r in items),'low_progress_alarms':sum(r['low_progress_alarm'] for r in items),'gripper_oscillation_alarms':sum(r['gripper_oscillation_alarm'] for r in items),'mechanisms':dict(Counter(r['weak_mechanism'] for r in items))}
 groups=defaultdict(list)
 for r in rows:groups[f"{r['suite']}:task{r['task_id']}"] .append(r)
 result={'label_status':'weak_mechanistic_audit_not_ground_truth','task_outcome_is_not_execution_mismatch_label':True,
         'overall':summarize(rows),'by_outcome':{'successful':summarize([r for r in rows if r['success'] is True]),
         'failed':summarize([r for r in rows if r['success'] is False])},
         'by_task':{k:summarize(v) for k,v in groups.items()},'episodes':rows}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps({'overall':result['overall'],'by_task':result['by_task']},indent=2))
if __name__=='__main__':main()
