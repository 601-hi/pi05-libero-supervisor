#!/usr/bin/env python3
"""Source-only post-run audit for reliability/sequence v1.1."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def ecdf(reference, values):
    reference = np.sort(np.asarray(reference, float))
    return (np.searchsorted(reference, values, side="right") + .5) / (len(reference) + 1.)


def run_episode(lr, reliability, unknown, decay, drift, threshold, clip=3.):
    risk = 0.; first = None
    for i, (value, weight, is_unknown) in enumerate(zip(lr, reliability, unknown)):
        if is_unknown:
            risk *= decay
        else:
            risk = max(0., decay * risk + weight * np.clip(value, -clip, clip) - drift)
        if risk >= threshold and first is None: first = i
    return first

def rank_auc(values, labels):
    values=np.asarray(values,float);labels=np.asarray(labels,bool);order=np.argsort(values,kind='mergesort');ranks=np.empty(len(values),float)
    i=0
    while i<len(values):
        j=i+1
        while j<len(values) and values[order[j]]==values[order[i]]:j+=1
        ranks[order[i:j]]=(i+1+j)/2;i=j
    n1=labels.sum();n0=(~labels).sum();return float((ranks[labels].sum()-n1*(n1+1)/2)/(n1*n0))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--calibration',type=Path,required=True);p.add_argument('--scores',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cal=json.loads(a.calibration.read_text(encoding='utf-8'));d=np.load(a.scores)
    lr=d['abnormal_logp']-d['normal_logp']; best=np.maximum(d['normal_logp'],d['abnormal_logp'])
    support=ecdf(cal['best_logp_reference'],best); stability=1-ecdf(cal['ensemble_std_reference'],d['ensemble_std']); reliability=np.minimum(support,stability)
    unknown=reliability<.2; known_n=(~unknown)&(lr<=cal['normal_lr_threshold']);known_a=(~unknown)&(lr>=cal['abnormal_lr_threshold']);amb=(~unknown)&~known_n&~known_a
    label=d['abnormal'].astype(bool)
    states={}
    for name,mask in [('normal',~label),('abnormal',label)]:
        states[name]={'steps':int(mask.sum()),'unknown_rate':float(unknown[mask].mean()),
                      'low_support_rate':float((support[mask]<.2).mean()),'low_stability_rate':float((stability[mask]<.2).mean()),
                      'known_normal_rate':float(known_n[mask].mean()),'ambiguous_rate':float(amb[mask].mean()),'known_abnormal_rate':float(known_a[mask].mean()),
                      'support_median':float(np.median(support[mask])),'stability_median':float(np.median(stability[mask])),'reliability_median':float(np.median(reliability[mask]))}
    scales={}
    # Original calibration has 19 episodes; synthetic episode offsets 19,38,57.
    for name,lo,hi in [('scale025',19,38),('scale050',38,57),('scale075',57,76)]:
        mask=label&(d['episode']>=lo)&(d['episode']<hi)
        scales[name]={'steps':int(mask.sum()),'lr_median':float(np.median(lr[mask])),'known_abnormal_rate':float(known_a[mask].mean()),'unknown_rate':float(unknown[mask].mean()),
                      'low_support_rate':float((support[mask]<.2).mean()),'low_stability_rate':float((stability[mask]<.2).mean())}
    episodes=[]
    for ep in np.unique(d['episode']):
        idx=np.flatnonzero(d['episode']==ep);idx=idx[np.argsort(d['action'][idx])]
        episodes.append((int(ep),bool(label[idx][0]),idx))
    candidates=[]
    for decay in (.8,.9,.95,.98):
      for drift in (0.,.1,.25,.5):
       for threshold in (2.,3.,4.,6.,8.):
        results=[(lab,run_episode(lr[idx],reliability[idx],unknown[idx],decay,drift,threshold)) for _,lab,idx in episodes]
        normal=[hit for lab,hit in results if not lab];abnormal=[hit for lab,hit in results if lab]
        candidates.append({'decay':decay,'drift':drift,'threshold':threshold,'normal_episode_false_alarm':float(np.mean([x is not None for x in normal])),'abnormal_episode_detection':float(np.mean([x is not None for x in abnormal])),'median_detection_step':float(np.median([x for x in abnormal if x is not None])) if any(x is not None for x in abnormal) else None})
    pareto=[]
    for limit in (0.,.1,.25,.5,1.):
        pool=[x for x in candidates if x['normal_episode_false_alarm']<=limit]
        chosen=max(pool,key=lambda x:(x['abnormal_episode_detection'],-(x['median_detection_step'] or 10**9)))
        pareto.append({'false_alarm_limit':limit,'best':chosen})
    selected=pareto[1]['best']; selected_by_scale={}
    for name,lo,hi in [('scale025',19,38),('scale050',38,57),('scale075',57,76)]:
        selected_episodes=[]
        for ep,lab,idx in episodes:
            if lo<=ep<hi:
                selected_episodes.append(run_episode(lr[idx],reliability[idx],unknown[idx],selected['decay'],selected['drift'],selected['threshold']))
        selected_by_scale[name]={'episodes':len(selected_episodes),'detection':float(np.mean([x is not None for x in selected_episodes])),
                                 'median_detection_step':float(np.median([x for x in selected_episodes if x is not None])) if any(x is not None for x in selected_episodes) else None}
    # More realistic source-only diagnostic: splice ten scaled-response scores
    # into the matching normal episode. The onset is deterministic and fixed
    # without looking at any target data.
    normal_episode_map={ep:idx for ep,lab,idx in episodes if not lab}
    localized=[]
    for scale_name,offset in [('scale025',19),('scale050',38),('scale075',57)]:
        for base_ep,nidx in normal_episode_map.items():
            aidx=next(idx for ep,lab,idx in episodes if ep==base_ep+offset)
            if not np.array_equal(d['action'][nidx],d['action'][aidx]): raise ValueError('paired source actions do not align')
            length=len(nidx); onset=min(30+(base_ep*37)%51,length-10)
            choose=nidx.copy();choose[onset:onset+10]=aidx[onset:onset+10]
            localized.append((scale_name,base_ep,onset,choose))
    window_records=[]
    for base_ep,nidx in normal_episode_map.items():
        onset=min(30+(base_ep*37)%51,len(nidx)-10);idx=nidx[onset:onset+10]
        window_records.append(('normal',False,float(np.mean(lr[idx])),float(np.sum(np.maximum(lr[idx],0))),float(np.mean(known_a[idx])),float(np.mean(unknown[idx])),float(np.mean(reliability[idx]*lr[idx]))))
    for scale_name,base_ep,onset,idx in localized:
        idx=idx[onset:onset+10]
        window_records.append((scale_name,True,float(np.mean(lr[idx])),float(np.sum(np.maximum(lr[idx],0))),float(np.mean(known_a[idx])),float(np.mean(unknown[idx])),float(np.mean(reliability[idx]*lr[idx]))))
    names=('mean_lr','positive_lr_sum','known_abnormal_fraction','unknown_fraction','weighted_mean_lr')
    window_auc={}
    for column,name in enumerate(names,2):
        window_auc[name]={'all_scales':rank_auc([r[column] for r in window_records],[r[1] for r in window_records]),
          **{scale:rank_auc([r[column] for r in window_records if r[0] in ('normal',scale)],[r[1] for r in window_records if r[0] in ('normal',scale)]) for scale in ('scale025','scale050','scale075')}}
    def rolling_mean(x,width=10):
        return np.convolve(x,np.ones(width)/width,mode='valid') if len(x)>=width else np.asarray([])
    normal_episode_max=np.asarray([rolling_mean(lr[idx]).max() for idx in normal_episode_map.values()])
    conformal_threshold=float(normal_episode_max.max())
    conformal_by_scale={}
    for scale_name in ('scale025','scale050','scale075'):
        hits=[];scores=[]
        for name,base_ep,onset,idx in localized:
            if name!=scale_name:continue
            score=float(np.mean(lr[idx[onset:onset+10]]));scores.append(score);hits.append(score>conformal_threshold)
        conformal_by_scale[scale_name]={'detections':int(sum(hits)),'episodes':len(hits),'rate':float(np.mean(hits)),'window_score_median':float(np.median(scores))}
    conformal={'window':10,'threshold_max_of_9_normal_episode_maxima':conformal_threshold,
      'finite_sample_note':'With 9 exchangeable calibration episodes, max-score split conformal has nominal next-episode exceedance bound 1/(9+1)=0.10 under exchangeability; this is not a target-domain guarantee.',
      'normal_episode_maxima':normal_episode_max.tolist(),'by_scale':conformal_by_scale}
    robust_conformal={}
    for beta in (0.,.5,1.,2.):
        robust=lr-beta*d['ensemble_std'];normal_max=np.asarray([rolling_mean(robust[idx]).max() for idx in normal_episode_map.values()]);threshold=float(normal_max.max());by={}
        for scale_name in ('scale025','scale050','scale075'):
            scores=[]
            for name,base_ep,onset,idx in localized:
                if name==scale_name:scores.append(float(np.mean(robust[idx[onset:onset+10]])))
            by[scale_name]={'rate':float(np.mean(np.asarray(scores)>threshold)),'median_score':float(np.median(scores))}
        robust_conformal[str(beta)]={'threshold':threshold,'normal_calibration_episode_false_alarms':int((normal_max>threshold).sum()),'by_scale':by}
    def has_unknown_run(values,length=3):
        run=0
        for value in values:
            run=run+1 if value else 0
            if run>=length:return True
        return False
    normal_protect=[has_unknown_run(unknown[idx]) for idx in normal_episode_map.values()]
    protective_by_scale={}
    for scale_name in ('scale025','scale050','scale075'):
        values=[]
        for name,base_ep,onset,idx in localized:
            if name==scale_name:values.append(has_unknown_run(unknown[idx[onset:onset+10]]))
        protective_by_scale[scale_name]={'episodes':len(values),'protective_observation_rate_during_window':float(np.mean(values))}
    protective={'unknown_patience':3,'normal_episode_protective_rate':float(np.mean(normal_protect)),'by_scale':protective_by_scale}
    localized_candidates=[]
    for decay in (.8,.9,.95,.98):
      for drift in (0.,.1,.25,.5):
       for threshold in (2.,3.,4.,6.,8.):
        normal_hits=[run_episode(lr[idx],reliability[idx],unknown[idx],decay,drift,threshold) for idx in normal_episode_map.values()]
        abnormal_hits=[]
        for scale_name,base_ep,onset,idx in localized:
            hit=run_episode(lr[idx],reliability[idx],unknown[idx],decay,drift,threshold)
            abnormal_hits.append((scale_name,None if hit is None or hit<onset or hit>=onset+10 else hit-onset))
        localized_candidates.append({'decay':decay,'drift':drift,'threshold':threshold,
          'normal_episode_false_alarm':float(np.mean([x is not None for x in normal_hits])),
          'localized_detection':float(np.mean([x is not None for _,x in abnormal_hits])),
          'median_delay':float(np.median([x for _,x in abnormal_hits if x is not None])) if any(x is not None for _,x in abnormal_hits) else None,
          'by_scale':{name:float(np.mean([x is not None for scale,x in abnormal_hits if scale==name])) for name in ('scale025','scale050','scale075')}})
    localized_pareto=[]
    for limit in (0.,.1,.25,.5,1.):
        pool=[x for x in localized_candidates if x['normal_episode_false_alarm']<=limit]
        chosen=max(pool,key=lambda x:(x['localized_detection'],-(x['median_delay'] or 10**9)))
        localized_pareto.append({'false_alarm_limit':limit,'best':chosen})
    result={'source_only_diagnostic':True,'states_by_class':states,'by_synthetic_scale':scales,'selected_sequence_by_scale':selected_by_scale,'pareto_under_current_formula':pareto,
            'localized_ten_step_diagnostic':{'onset_rule':'30 + (base_episode * 37) mod 51, clipped to length-10','window_auc':window_auc,'episode_max_conformal':conformal,'robust_lcb_conformal':robust_conformal,'protective_unknown':protective,'pareto':localized_pareto},
            'interpretation_limits':['synthetic abnormal spans the entire episode, so detection step is not delay from a localized onset','v1.1 target data was not used']}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
