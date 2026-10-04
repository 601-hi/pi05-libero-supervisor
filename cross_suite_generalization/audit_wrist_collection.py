#!/usr/bin/env python3
"""Audit completed fixed-noise wrist development traces and paired causality."""
from __future__ import annotations
import argparse, collections, json, re
from pathlib import Path
import numpy as np


def load(path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    rows=[json.loads(line) for line in path.open(encoding='utf-8') if line.strip()]
    return ([r for r in rows if r.get('event')=='step'],[r for r in rows if r.get('event')=='inference'],
            [r for r in rows if r.get('event')=='episode_end'])


def max_difference(a: list[dict], b: list[dict], key: str, stop: int) -> float:
    if stop <= 0:return 0.0
    x=np.asarray([r[key] for r in a[:stop]],float);y=np.asarray([r[key] for r in b[:stop]],float)
    return float(np.max(np.abs(x-y)))


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    completed={};trace_reports=[];errors=[]
    pattern=re.compile(r'(?P<suite>libero_(?:object|goal|90))_task(?P<task>\d+)_seed31_(?P<tag>normal|scale075|scale050)$')
    for trace in sorted(a.root.glob('*.jsonl')):
        match=pattern.fullmatch(trace.stem)
        if not match:continue
        steps,inferences,ends=load(trace)
        if len(ends)!=1:continue  # A currently written trace is not yet complete.
        suite,task,tag=match['suite'],int(match['task']),match['tag']; expected_scale={'normal':1.,'scale075':.75,'scale050':.5}[tag]
        expected_active=0 if tag=='normal' else 10
        visual_files=list((a.root/'visuals'/trace.stem).glob('*.npz'))
        row={'trace':str(trace),'suite':suite,'task':task,'tag':tag,'steps':len(steps),'inferences':len(inferences),
             'success':bool(ends[0]['success']),'active_steps':sum(bool(r['disturbance_active']) for r in steps),
             'visual_files':len(visual_files)}
        if row['active_steps']!=expected_active:errors.append(f'{trace.name}: active steps {row["active_steps"]} != {expected_active}')
        if not all(r.get('sampling_noise_seed')==2026090531 for r in inferences):errors.append(f'{trace.name}: noise seed mismatch')
        if [int(r['action_index']) for r in steps]!=list(range(len(steps))):errors.append(f'{trace.name}: non-contiguous actions')
        if len(visual_files)!=1:errors.append(f'{trace.name}: expected one visual NPZ')
        else:
            v=np.load(visual_files[0],allow_pickle=False)
            row['agent_frames']=len(v['agent_images']);row['wrist_frames']=len(v['wrist_images'])
            if not (len(steps)==row['agent_frames']==row['wrist_frames']):errors.append(f'{trace.name}: step/frame mismatch')
            if not np.array_equal(v['action_indices'],np.arange(len(steps))):errors.append(f'{trace.name}: visual indices mismatch')
        for r in steps:
            intended=np.asarray(r['intended_action'],float);executed=np.asarray(r['executed_action'],float)
            expected=intended.copy()
            if r['disturbance_active']:expected[:3]*=expected_scale
            if not np.array_equal(executed,expected):errors.append(f'{trace.name}: executed action mismatch at {r["action_index"]}');break
        trace_reports.append(row);completed[(suite,task,tag)]=steps
    pair_reports=[]
    for suite,task in sorted({(key[0],key[1]) for key in completed}):
        if (suite,task,'normal') not in completed:continue
        normal=completed[(suite,task,'normal')]
        for tag in ('scale075','scale050'):
            key=(suite,task,tag)
            if key not in completed:continue
            disturbed=completed[key];active=[i for i,r in enumerate(disturbed) if r['disturbance_active']]
            onset=active[0]; common=min(onset,len(normal),len(disturbed))
            report={'suite':suite,'task':task,'tag':tag,'onset':onset,
              'pre_intended_max_diff':max_difference(normal,disturbed,'intended_action',common),
              'pre_executed_max_diff':max_difference(normal,disturbed,'executed_action',common),
              'pre_eef_before_max_diff':max_difference(normal,disturbed,'eef_pos_before',common),
              'pre_eef_after_max_diff':max_difference(normal,disturbed,'eef_pos_after',common)}
            pair_reports.append(report)
            if any(report[k]!=0 for k in report if k.endswith('max_diff')):errors.append(f'{suite} task{task} {tag}: pre-onset pair mismatch')
            if onset>=len(normal) or not np.array_equal(np.asarray(normal[onset]['intended_action']),np.asarray(disturbed[onset]['intended_action'])):
                errors.append(f'{suite} task{task} {tag}: onset intended action mismatch')
    result={'complete_traces':len(trace_reports),'trace_reports':trace_reports,'pair_reports':pair_reports,'errors':errors,'passed':not errors}
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({'complete_traces':len(trace_reports),'pairs':len(pair_reports),'errors':errors,'passed':not errors},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
