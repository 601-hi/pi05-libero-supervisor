#!/usr/bin/env python3
"""Calibrate/evaluate visual rescue of ambiguous trajectory states on seed20."""
from __future__ import annotations
import argparse,collections,json
from pathlib import Path
import numpy as np
import torch
from train_visual_ambiguity_verifier import VisualVerifier,states


def second_of_three(values):
    out=np.full(len(values),-np.inf)
    for i in range(1,len(values)):
        w=np.asarray(values[max(0,i-2):i+1]); out[i]=np.partition(w,-2)[-2]
    return out


def main():
    p=argparse.ArgumentParser(); p.add_argument('--visual',type=Path,required=True); p.add_argument('--scores',type=Path,required=True)
    p.add_argument('--fusion',type=Path,required=True);p.add_argument('--visual-model',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--max-normal-episode-fp',type=int,default=0); a=p.parse_args()
    v=np.load(a.visual,allow_pickle=False);s=np.load(a.scores,allow_pickle=False);c=json.loads(a.fusion.read_text(encoding='utf-8'))
    ck=torch.load(a.visual_model,map_location='cpu');m=VisualVerifier(int(ck['input_dim']));m.load_state_dict(ck['state_dict']);m.eval()
    x=np.c_[v['visual'],v['x_condition'][:,:8],v['x_condition'][:,58:68]]
    with torch.no_grad(): prob=torch.sigmoid(m(torch.tensor((x-ck['mean'])/ck['scale'],dtype=torch.float32))).numpy()
    state=states(s,c); valid=v['valid'].astype(bool); clean=np.isclose(v['scale'],1);active=v['abnormal'].astype(bool)
    novelty=np.zeros(len(state))
    for i in range(len(state)):
        tc=c['by_task'][str(int(v['task_id'][i]))]
        novelty[i]=(tc['normal_logp_center']-s['normal_logp'][i])/tc['normal_logp_scale']
    groups=collections.defaultdict(list)
    for i,e in enumerate(v['episode_id']):groups[str(e)].append(i)
    normal_triggers=[]
    for eid,ix in groups.items():
        ix=sorted(ix,key=lambda i:int(v['action_index'][i])); vals=np.where(valid[ix]&(state[ix]=='ambiguous_overlap'),prob[ix],-np.inf)
        if clean[ix].all():normal_triggers.append(float(np.max(second_of_three(vals))))
    candidates=np.unique(np.r_[normal_triggers,[np.nextafter(q,np.inf) for q in normal_triggers],np.inf])
    choices=[]
    for threshold in candidates:
        fp=det=recall=0
        for eid,ix in groups.items():
            ix=sorted(ix,key=lambda i:int(v['action_index'][i])); visual_point=valid[ix]&(state[ix]=='ambiguous_overlap')&(prob[ix]>=threshold)
            base_point=(state[ix]=='known_abnormal')|((state[ix]=='unknown_abnormal')&(novelty[ix]>=c['unknown_novelty_threshold']))
            combined=base_point|visual_point; alarm=np.zeros(len(ix),bool)
            for j in range(len(ix)):alarm[j]=combined[max(0,j-2):j+1].sum()>=2
            if clean[ix].all():fp+=bool(alarm.any())
            elif active[ix].any():det+=bool(np.any(alarm&active[ix]));recall+=int(np.sum(alarm&active[ix]))
        if fp<=a.max_normal_episode_fp:choices.append((det,recall,-fp,float(threshold)))
    det,recall,negfp,threshold=max(choices)
    details=[];total_active=0;inactive_alarm=active_alarm=inactive_count=0
    for eid,ix in groups.items():
        ix=sorted(ix,key=lambda i:int(v['action_index'][i])); visual_point=valid[ix]&(state[ix]=='ambiguous_overlap')&(prob[ix]>=threshold)
        base_point=(state[ix]=='known_abnormal')|((state[ix]=='unknown_abnormal')&(novelty[ix]>=c['unknown_novelty_threshold']));combined=base_point|visual_point;alarm=np.zeros(len(ix),bool)
        for j in range(len(ix)):alarm[j]=combined[max(0,j-2):j+1].sum()>=2
        aa=active[ix];total_active+=int(aa.sum());active_alarm+=int(np.sum(alarm&aa));inactive_alarm+=int(np.sum(alarm&~aa));inactive_count+=int((~aa).sum())
        details.append({'episode_id':eid,'task_id':int(v['task_id'][ix[0]]),'scale':float(v['scale'][ix[0]]),'active_steps':int(aa.sum()),
          'detected':bool(np.any(alarm&aa)),'any_alarm':bool(alarm.any()),'visual_rescue_steps':int(visual_point.sum())})
    report={'threshold':threshold,'normal_episode_budget':a.max_normal_episode_fp,'normal_episodes':sum(d['active_steps']==0 for d in details),
      'normal_false_alarms':sum(d['any_alarm'] for d in details if d['active_steps']==0),'abnormal_episodes':sum(d['active_steps']>0 for d in details),
      'abnormal_detections':sum(d['detected'] for d in details if d['active_steps']>0),'active_step_recall':active_alarm/max(total_active,1),
      'inactive_step_alarm_rate':inactive_alarm/max(inactive_count,1),'details':details}
    a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k!='details'},indent=2))
if __name__=='__main__':main()
