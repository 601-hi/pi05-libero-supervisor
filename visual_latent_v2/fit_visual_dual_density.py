"""Select a visual dual-density candidate using calibration episodes only."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from visual_dual_density import fit,save_model,score


def evaluate(data, values, base, budget):
    ambiguous=data['previous_ambiguous_any'].astype(bool);active=data['previous_active_any'].astype(bool)
    groups={eid:np.flatnonzero(data['episode_id']==eid) for eid in np.unique(data['episode_id'])}
    by={r['episode_id']:r for r in base['details']};normal_max=[]
    for eid,ix in groups.items():
        if not by[eid]['has_active_fault']:
            v=values[ix][ambiguous[ix]];normal_max.append(float(v.max()) if len(v) else -np.inf)
    candidates=np.unique(np.r_[normal_max,[np.nextafter(v,np.inf) for v in normal_max],np.inf]);choices=[]
    for threshold in candidates:
        fp=det=rescue=active_hits=inactive=0
        for eid,ix in groups.items():
            trigger=ambiguous[ix]&(values[ix]>=threshold);b=by[eid]
            if not b['has_active_fault']:fp+=bool(b['any_alarm'] or trigger.any())
            else:
                va=bool(np.any(trigger&active[ix]));det+=bool(b['detected_during_active'] or va);rescue+=bool(va and not b['detected_during_active'])
                active_hits+=int(np.sum(trigger&active[ix]));inactive+=int(np.sum(trigger&~active[ix]))
        if fp<=budget:choices.append((det,rescue,active_hits,-inactive,-fp,float(threshold)))
    return max(choices)


def main():
    p=argparse.ArgumentParser();p.add_argument('--train-data',type=Path,required=True);p.add_argument('--train-features',type=Path,required=True)
    p.add_argument('--cal-data',type=Path,required=True);p.add_argument('--cal-features',type=Path,required=True);p.add_argument('--base',type=Path,required=True)
    p.add_argument('--budget',type=int,default=1);p.add_argument('--out-model',type=Path,required=True);p.add_argument('--out-config',type=Path,required=True);a=p.parse_args()
    td=np.load(a.train_data);tx=np.load(a.train_features)['features'];cd=np.load(a.cal_data);cx=np.load(a.cal_features)['features'];base=json.loads(a.base.read_text())
    reports=[];best=None
    for dim in (8,16,32):
      for shrink in (5.0,10.0,20.0):
        model=fit(td,tx,dim,shrink);v=score(model,cd,cx);result=evaluate(cd,v,base,a.budget)
        row={'pca_dim':dim,'shrinkage':shrink,'detections':result[0],'rescues':result[1],'active_hits':result[2],
             'inactive_alarms':-result[3],'normal_fp':-result[4],'threshold':result[5]};reports.append(row)
        key=(result[0],result[1],result[2],result[3],result[4],-dim,-shrink)
        if best is None or key>best[0]:best=(key,model,row)
    save_model(a.out_model,best[1]);config={'selected':best[2],'candidates':reports,'budget':a.budget,
      'warning':'Exploratory v2 selected on seed20; seed21 is diagnostic only because it was already opened for v1.'}
    a.out_config.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(config,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
