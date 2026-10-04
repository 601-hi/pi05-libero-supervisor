"""Audit the legacy empirical EEF-to-pixel fit without refitting it."""
import argparse, json
from collections import Counter
from pathlib import Path
import numpy as np


def stats(values):
    a=np.asarray(values,float)
    return {'n':len(a),'median':float(np.median(a)),'p90':float(np.percentile(a,90)),
            'max':float(np.max(a)),'mean':float(np.mean(a))}


def main():
    p=argparse.ArgumentParser();p.add_argument('--fit',type=Path,required=True)
    p.add_argument('--pixels',type=Path,required=True);p.add_argument('--pairs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    fit=json.loads(a.fit.read_text(encoding='utf-8'))
    pixels={r['anonymous_id']:r for r in json.loads(a.pixels.read_text(encoding='utf-8'))['records']}
    pairs={r['anonymous_id']:r for r in json.loads(a.pairs.read_text(encoding='utf-8'))['records']}
    best=fit['best_model']; candidate=next(r for r in fit['all_candidates']
        if r['model']==best['model'] and r['ridge']==best['ridge'])
    errors={r['anonymous_id']:r['error_px'] for r in candidate['per_episode']}
    by_conf={c:stats([errors[i] for i,v in pixels.items() if v['confidence']==c])
             for c in sorted(set(v['confidence'] for v in pixels.values()))}
    xyz=np.asarray([pairs[i]['eef_position_xyz'] for i in errors])
    xy=np.asarray([pixels[i]['gripper_center_xy'] for i in errors])
    p90=best['loo_p90_error_px']
    out={'sample_count':len(errors),'annotation_definition':json.loads(a.pixels.read_text(encoding='utf-8')).get('definition'),
      'confidence_counts':dict(Counter(v['confidence'] for v in pixels.values())),
      'loo_error_all':stats(list(errors.values())),'loo_error_by_annotation_confidence':by_conf,
      'empirical_coverage_at_reported_p90':float(np.mean(np.asarray(list(errors.values()))<=p90)),
      'points_exceeding_reported_p90':sum(v>p90 for v in errors.values()),
      'eef_xyz_range_m':{'min':xyz.min(0).tolist(),'max':xyz.max(0).tolist()},
      'annotated_xy_range_px':{'min':xy.min(0).tolist(),'max':xy.max(0).tolist()},
      'projection_domain_present':'projection_domain' in fit,
      'orientation_used':False,'gripper_opening_used':False,
      'model_selection_and_reporting_same_samples':True,
      'hard_safety_bound_supported':False,
      'limitations':['Only one manually selected close frame per episode','Three low-confidence labels are weighted equally',
        'Annotation was made on a 16-pixel grid','Candidate model and ridge selected on the same 20 LOO errors used for reporting',
        'No independent calibration/test split','No EEF orientation or finger opening','No bound to acquisition domain']}
    a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
