#!/usr/bin/env python3
"""Diagnostic-only target evaluation of source-frozen robust window rule."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
def main():
 p=argparse.ArgumentParser();p.add_argument('--scores',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--threshold',type=float,default=2.6228122472763062);p.add_argument('--beta',type=float,default=1.);p.add_argument('--window',type=int,default=10);a=p.parse_args();d=np.load(a.scores);robust=d['abnormal_logp']-d['normal_logp']-a.beta*d['ensemble_std'];episodes=[]
 for fi,ep in sorted(set(zip(d['file_index'],d['episode_idx']))):
  mask=(d['file_index']==fi)&(d['episode_idx']==ep);idx=np.flatnonzero(mask)[np.argsort(d['action_index'][mask])];scores=robust[idx];actions=d['action_index'][idx];active=d['abnormal'][idx].astype(bool);rolling=np.convolve(scores,np.ones(a.window)/a.window,mode='valid');alarm_actions=actions[a.window-1:][rolling>a.threshold];normal=not active.any();active_hits=alarm_actions[np.isin(alarm_actions,actions[active])]
  episodes.append({'file_index':int(fi),'episode_idx':int(ep),'normal':normal,'any_alarm':bool(len(alarm_actions)),'active_detection':bool(len(active_hits)),'delay':int(active_hits.min()-actions[active].min()) if len(active_hits) else None})
 normal=[x for x in episodes if x['normal']];abnormal=[x for x in episodes if not x['normal']];result={'diagnostic_only':True,'source_frozen':{'threshold':a.threshold,'beta':a.beta,'window':a.window},'normal_episode_false_alarm':float(np.mean([x['any_alarm'] for x in normal])) if normal else None,'abnormal_episode_detection':float(np.mean([x['active_detection'] for x in abnormal])) if abnormal else None,'median_delay':float(np.median([x['delay'] for x in abnormal if x['delay'] is not None])) if any(x['delay'] is not None for x in abnormal) else None,'normal_episodes':len(normal),'abnormal_episodes':len(abnormal),'details':episodes}
 a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='details'},indent=2))
if __name__=='__main__':main()
