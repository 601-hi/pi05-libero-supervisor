#!/usr/bin/env python3
"""Evaluate frozen transfer scores without choosing thresholds from abnormal data."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def auc_rank(labels, scores):
    labels = np.asarray(labels, bool); scores = np.asarray(scores, float)
    order = np.argsort(scores, kind="mergesort"); ranks = np.empty(len(scores), float)
    sorted_scores = scores[order]; start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]: stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop); start = stop
    p = int(labels.sum()); n = len(labels) - p
    return float((ranks[labels].sum() - p * (p + 1) / 2) / (p * n))


def causal_mean(score, file_id, task, episode, action, width=3):
    out = np.empty_like(score, dtype=float)
    groups = {}
    for i, key in enumerate(zip(file_id, task, episode)):
        groups.setdefault(key, []).append(i)
    for indices in groups.values():
        indices.sort(key=lambda i: action[i])
        for position, index in enumerate(indices):
            history = indices[max(0, position - width + 1):position + 1]
            out[index] = np.mean(score[history])
    return out


def main():
    p=argparse.ArgumentParser(); p.add_argument('--scores',type=Path,required=True)
    p.add_argument('--normal-reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    data=np.load(a.scores); reference=np.load(a.normal_reference)
    label=data['abnormal'].astype(bool); score=data['score'].astype(float)
    smooth=causal_mean(score,data['file_index'],data['task_id'],data['episode_idx'],data['action_index'])
    ref_score=reference['score'].astype(float)
    ref_smooth=causal_mean(ref_score,reference['file_index'],reference['task_id'],reference['episode_idx'],reference['action_index'])
    result={'rows':int(len(score)),'abnormal_rows':int(label.sum()),'single_step_auc':auc_rank(label,score),
            'causal_mean_3_auc':auc_rank(label,smooth),'thresholds':{}}
    for q in (95,99):
        threshold=float(np.percentile(ref_smooth,q)); alarm=smooth>threshold
        episode_hits=[]
        for key in set(zip(data['file_index'],data['task_id'],data['episode_idx'])):
            mask=np.asarray([x==key for x in zip(data['file_index'],data['task_id'],data['episode_idx'])])
            active=mask & label
            if active.any(): episode_hits.append(bool((alarm & active).any()))
        result['thresholds'][f'normal_q{q}']={'value':threshold,
            'active_step_recall':float(alarm[label].mean()),
            'inactive_step_alarm_rate':float(alarm[~label].mean()),
            'abnormal_episode_detection':float(np.mean(episode_hits)),
            'abnormal_episodes':len(episode_hits)}
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
