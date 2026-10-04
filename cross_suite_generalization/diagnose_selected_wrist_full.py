"""Generate label-free stepwise wrist attachment events from full tracks."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--tracks',type=Path,required=True);p.add_argument('--thresholds',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--settle',type=int,default=5);p.add_argument('--confirm',type=int,default=3);p.add_argument('--grace',type=int,default=20);a=p.parse_args()
    records=json.loads(a.tracks.read_text(encoding='utf-8'))['records'];t=json.loads(a.thresholds.read_text(encoding='utf-8'));rows=[]
    for r in records:
        if r.get('status')!='tracked':rows.append({'anonymous_id':r['anonymous_id'],'status':r.get('status'),'events':[]});continue
        xy=np.asarray([[np.nan,np.nan] if x is None else x for x in r['centroids_xy']],float)/224.0;area=np.asarray(r['area_fraction'],float)
        speed=np.r_[np.nan,np.linalg.norm(np.diff(xy,axis=0),axis=1)];change=np.r_[np.nan,np.abs(np.diff(np.log(np.maximum(area,1e-8))))]
        stable=np.isfinite(speed)&np.isfinite(change)&(area>=t['minimum_visible_area_q01'])&(speed<=t['speed_threshold_q95'])&(change<=t['area_log_change_threshold_q95'])
        established=None;events=[];instability=None
        for i in range(a.settle,len(stable)):
            if established is None and i>=a.settle+a.confirm-1 and stable[i-a.confirm+1:i+1].all():established=i;events.append({'relative_frame':i,'type':'attachment_established'})
            if established is not None and i>=established+a.confirm and (~stable[i-a.confirm+1:i+1]).all():
                instability=i;events.append({'relative_frame':i,'type':'attachment_instability'});break
        if instability is not None:
            for j in range(instability+a.confirm,len(stable)):
                if stable[j-a.confirm+1:j+1].all():
                    events.append({'relative_frame':j,'type':'attachment_recovered','latency_from_instability':j-instability});break
        if established is None and len(stable)>a.grace:events.append({'relative_frame':min(a.grace,len(stable)-1),'type':'grasp_not_established'})
        rows.append({'anonymous_id':r['anonymous_id'],'status':'diagnosed','close_frame':r['close_frame'],'tracked_frames':r['tracked_frames'],'attachment_established_frame':established,'events':events,'stable_fraction':float(stable[a.settle:].mean()) if len(stable)>a.settle else None})
    result={'schema_version':1,'labels_used':False,'threshold_source':str(a.thresholds),'summary':{'episodes':len(rows),'established':sum(x.get('attachment_established_frame') is not None for x in rows),'not_established':sum(any(e['type']=='grasp_not_established' for e in x['events']) for x in rows),'instability':sum(any(e['type']=='attachment_instability' for e in x['events']) for x in rows),'recovered_after_instability':sum(any(e['type']=='attachment_recovered' for e in x['events']) for x in rows)},'records':rows}
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result['summary'],indent=2))
if __name__=='__main__':main()
