"""Fit wrist-relative attachment stability thresholds on development labels."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--predictions',type=Path,required=True);p.add_argument('--annotations',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--settle',type=int,default=5);a=p.parse_args()
    records=json.loads(a.predictions.read_text(encoding='utf-8'))['records'];annotations=json.loads(a.annotations.read_text(encoding='utf-8'))['annotations']
    speeds=[];area_changes=[];areas=[]
    for r in records:
        for cid in annotations[r['anonymous_id']]['acceptable_candidate_ids']:
            key=str(cid)
            if key not in r['post_centroids_xy']:continue
            xy=np.asarray([[np.nan,np.nan] if x is None else x for x in r['post_centroids_xy'][key]],float)/224.0
            area=np.asarray(r['post_area_fraction'][key],float)
            speed=np.linalg.norm(np.diff(xy,axis=0),axis=1)[a.settle:]
            change=np.abs(np.diff(np.log(np.maximum(area,1e-8))))[a.settle:]
            speeds.extend(speed[np.isfinite(speed)]);area_changes.extend(change[np.isfinite(change)]);areas.extend(area[a.settle:][np.isfinite(area[a.settle:])])
    result={'schema_version':1,'development_only':True,'settle_frames':a.settle,'speed_normalized_by_image_width':True,
            'speed_threshold_q95':float(np.quantile(speeds,.95)),'area_log_change_threshold_q95':float(np.quantile(area_changes,.95)),
            'minimum_visible_area_q01':float(np.quantile(areas,.01)),'counts':{'speed':len(speeds),'area_change':len(area_changes),'area':len(areas)}}
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
