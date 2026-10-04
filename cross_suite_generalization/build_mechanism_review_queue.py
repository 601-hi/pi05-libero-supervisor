#!/usr/bin/env python3
"""Build an auditable review queue from weak-mechanism diagnostics."""
from __future__ import annotations
import argparse,json
from pathlib import Path
def k(e):return (Path(e['path']).name,int(e['task_id']),int(e['episode_idx']))
def main():
 p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--execution',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 audit=json.loads(a.audit.read_text(encoding='utf-8'));execution=json.loads(a.execution.read_text(encoding='utf-8'));ex={k(e):e for e in execution['episodes']};items=[]
 for row in audit['episodes']:
  match=next((e for key,e in ex.items() if key[1:]==(row['task_id'],row['episode_idx']) and row['suite'] in key[0]),None)
  if not match:continue
  if row['success'] is False and row['weak_mechanism']=='outcome_or_policy_failure_without_motion_alarm':priority=1;reason='failed_without_motion_evidence'
  elif row['success'] is True and row['execution_candidate_alarm']:priority=2;reason='execution_false_alarm_candidate'
  elif row['success'] is False and not row['execution_candidate_alarm']:priority=3;reason='execution_miss_candidate'
  else:continue
  first=match['temporal_rules']['1of1']['first_alarm_action'];center=first if first is not None else max(0,match['steps']-30)
  items.append({'priority':priority,'reason':reason,'trace':match['path'],'suite':row['suite'],'task_id':row['task_id'],'episode_idx':row['episode_idx'],'success':row['success'],'review_action_range':[max(0,center-15),min(match['steps']-1,center+15)],'weak_signals':{'execution':row['execution_candidate_alarm'],'low_progress':row['low_progress_alarm'],'gripper_oscillation':row['gripper_oscillation_alarm']},'label_policy':'human_review_required_not_training_label'})
 items.sort(key=lambda x:(x['priority'],x['suite'],x['task_id'],x['episode_idx']));result={'schema_version':1,'items':items,'counts':{str(i):sum(x['priority']==i for x in items) for i in (1,2,3)}}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result['counts'],indent=2))
if __name__=='__main__':main()
