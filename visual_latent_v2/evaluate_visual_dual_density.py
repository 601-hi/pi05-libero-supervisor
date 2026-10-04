"""Evaluate a fixed visual dual-density model and threshold."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from visual_dual_density import load_model,score


def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
 p.add_argument('--config',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 d=np.load(a.data);x=np.load(a.features)['features'];m=load_model(a.model);v=score(m,d,x);cfg=json.loads(a.config.read_text());threshold=float(cfg['selected']['threshold'])
 ambiguous=d['previous_ambiguous_any'].astype(bool);active=d['previous_active_any'].astype(bool);groups={eid:np.flatnonzero(d['episode_id']==eid) for eid in np.unique(d['episode_id'])}
 base=json.loads(a.base.read_text());by={r['episode_id']:r for r in base['details']};details=[]
 for eid,ix in groups.items():
  trigger=ambiguous[ix]&(v[ix]>=threshold);b=by[eid];va=bool(np.any(trigger&active[ix]));details.append({'episode_id':str(eid),'task_id':int(d['task_id'][ix[0]]),'scale':float(d['scale'][ix[0]]),
   'has_active_fault':bool(b['has_active_fault']),'base_detected':bool(b['detected_during_active']),'base_any_alarm':bool(b['any_alarm']),
   'visual_active':va,'visual_any':bool(trigger.any()),'cascade_detected':bool(b['detected_during_active'] or va),'cascade_any':bool(b['any_alarm'] or trigger.any()),
   'active_hits':int(np.sum(trigger&active[ix])),'inactive_hits':int(np.sum(trigger&~active[ix])),'max_active_score':float(v[ix][active[ix]].max()) if active[ix].any() else None})
 normal=[r for r in details if not r['has_active_fault']];abnormal=[r for r in details if r['has_active_fault']]
 report={'threshold':threshold,'normal_episodes':len(normal),'abnormal_episodes':len(abnormal),'base_normal_fp':sum(r['base_any_alarm'] for r in normal),
  'cascade_normal_fp':sum(r['cascade_any'] for r in normal),'base_detections':sum(r['base_detected'] for r in abnormal),'cascade_detections':sum(r['cascade_detected'] for r in abnormal),
  'visual_rescues':sum(r['cascade_detected'] and not r['base_detected'] for r in abnormal),'active_hits':sum(r['active_hits'] for r in abnormal),'inactive_hits':sum(r['inactive_hits'] for r in details),
  'details':details,'warning':'seed21 diagnostic only; not an independent v2 test'}
 a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k!='details'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
