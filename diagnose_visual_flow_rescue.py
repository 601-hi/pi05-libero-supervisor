#!/usr/bin/env python3
"""Explain why aligned flow evidence does or does not rescue missed episodes."""
from __future__ import annotations
import argparse, collections, json
from pathlib import Path
import numpy as np
from evaluate_visual_flow_cascade import classify_states, persistent


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--visual',type=Path,required=True); p.add_argument('--visual-scores',type=Path,required=True)
    p.add_argument('--dynamics-scores',type=Path,required=True); p.add_argument('--fusion',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); v=np.load(a.visual,allow_pickle=False); f=np.load(a.visual_scores,allow_pickle=False); d=np.load(a.dynamics_scores,allow_pickle=False)
    c=json.loads(a.fusion.read_text(encoding='utf-8')); state=classify_states(d,c,v['task_id']); active=v['abnormal'].astype(bool); valid=v['valid'].astype(bool)
    evidence=-f['directed_response']; novelty=np.zeros(len(state))
    for i,t in enumerate(v['task_id']):
        z={**c,**c.get('by_task',{}).get(str(int(t)),{})}; novelty[i]=(z['normal_logp_center']-d['normal_logp'][i])/z['normal_logp_scale']
    groups=collections.defaultdict(list)
    for i,e in enumerate(v['episode_id']): groups[str(e)].append(i)
    rows=[]
    for eid,ix0 in groups.items():
        ix=np.asarray(sorted(ix0,key=lambda i:int(v['action_index'][i]))); aa=active[ix]
        if not aa.any(): continue
        base=(state[ix]=='known_abnormal')|((state[ix]=='unknown_abnormal')&(novelty[ix]>=c['unknown_novelty_threshold']))
        missed=not np.any(persistent(base)&aa)
        if not missed: continue
        av=aa&valid[ix]; amb=av&(state[ix]=='ambiguous_overlap'); ev=evidence[ix]
        rows.append({'episode_id':eid,'task_id':int(v['task_id'][ix[0]]),'scale':float(v['scale'][ix[0]]),
          'active_valid_steps':int(av.sum()),'active_ambiguous_steps':int(amb.sum()),
          'max_active_visual_evidence':float(np.max(ev[av])) if av.any() else None,
          'max_ambiguous_active_visual_evidence':float(np.max(ev[amb])) if amb.any() else None,
          'active_state_counts':{s:int(np.sum(aa&(state[ix]==s))) for s in np.unique(state[ix])}})
    result={'missed_episode_count':len(rows),'missed':rows}
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
