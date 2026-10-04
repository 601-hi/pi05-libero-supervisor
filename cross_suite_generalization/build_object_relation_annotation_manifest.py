#!/usr/bin/env python3
"""Build frame-level object-relation annotation jobs from audited sidecars."""
from __future__ import annotations
import argparse,json,re
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--goal-map',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--stride',type=int,default=20);a=p.parse_args();audit=json.loads(a.audit.read_text(encoding='utf-8'));goals=json.loads(a.goal_map.read_text(encoding='utf-8'));jobs=[]
 for r in audit['records']:
  if r['mapping_status']!='unique':continue
  c=r['candidate_traces'][0];n=r['frames'];indices=sorted(set([0,n-1,*range(0,n,a.stride)]))
  goal_key=f"{c['suite']}:task{r['key']['task_id']}"
  jobs.append({'sidecar':r['path'],'suite':c['suite'],'trace':c['trace'],'task_id_for_data_join_only':r['key']['task_id'],'episode_idx':r['key']['episode_idx'],'outcome_for_audit_only':r['key']['outcome'],'goal_language':goals[goal_key],'frame_indices':indices,
   'annotation_fields':{'object_visible':'unknown','target_visible':'unknown','gripper_object_contact':'unknown','object_in_target':'unknown','release_event':'unknown','occluded':'unknown','annotator_confidence':'unknown'},
   'feature_firewall':{'model_inputs_allowed':['agent_image','wrist_image','goal_language'],'model_inputs_forbidden':['task_id','suite','reward','success','disturbance_label','eef_pose','joint_state']}})
 result={'schema_version':1,'jobs':jobs,'unique_episodes':len(jobs),'frames_to_annotate':sum(len(x['frame_indices']) for x in jobs),'excluded_ambiguous_sidecars':audit['ambiguous_or_unmatched'],'label_semantics':'frame-level observable relations; episode outcome is audit metadata only'}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='jobs'},indent=2))
if __name__=='__main__':main()
