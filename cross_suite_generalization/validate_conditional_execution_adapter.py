#!/usr/bin/env python3
"""Numerically compare the online adapter with the frozen offline scorer."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from vla_supervisor.conditional_execution import ConditionalDualExpertScorer

def main():
    p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--model',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=[json.loads(x) for x in a.trace.open(encoding='utf-8') if x.strip()]
    steps=[r for r in rows if r.get('event')=='step']
    scorer=ConditionalDualExpertScorer(a.model); observed=[]
    history=[]
    for r in steps:
        before={'eef_pos':r['eef_pos_before'],'joint_pos':r['joint_pos_before'],'gripper_qpos':r['gripper_qpos_before']}
        after={'eef_pos':r['eef_pos_after'],'joint_pos':r['joint_pos_after'],'gripper_qpos':r['gripper_qpos_after']}
        _,_,e=scorer(r['intended_action'],before,after,history)
        if r['action_index'] >= 5: observed.append(e)
        history.append({'intended_action':r['intended_action']})
    ref=np.load(a.reference)
    checks={
      'normal_logp_mean':np.asarray([e['normal_logp_mean'] for e in observed]),
      'abnormal_logp_mean':np.asarray([e['abnormal_logp_mean'] for e in observed]),
      'likelihood_ratio_mean':np.asarray([e['likelihood_ratio_mean'] for e in observed]),
      'ensemble_std':np.asarray([e['ensemble_std'] for e in observed]),
    }
    expected={'normal_logp_mean':ref['normal_logp'],'abnormal_logp_mean':ref['abnormal_logp'],
              'likelihood_ratio_mean':ref['abnormal_logp']-ref['normal_logp'],'ensemble_std':ref['ensemble_std']}
    errors={k:float(np.max(np.abs(v-expected[k]))) for k,v in checks.items()}
    tolerance=1e-3
    result={'status':'PASS' if max(errors.values())<tolerance else 'FAIL','rows':len(observed),
            'absolute_tolerance':tolerance,'max_absolute_errors':errors,
            'feature_leakage_prohibited':['task_id','reward','success','disturbance_active','object_state']}
    assert result['status']=='PASS',result
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
