"""One-shot evaluation of frozen action/visual/fusion baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import expit


def auc(scores, labels):
    positive = scores[labels == 1]; negative = scores[labels == 0]
    return float(sum((a > b) + .5 * (a == b) for a in positive for b in negative) / (len(positive) * len(negative))) if len(positive) and len(negative) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen=json.loads(args.frozen.read_text(encoding="utf-8")); manifest=json.loads(args.manifest.read_text(encoding="utf-8")); tracks=json.loads(args.tracks.read_text(encoding="utf-8"))
    meta={r["episode_id"]:r for r in manifest["episodes"]}; measured={(r["episode_id"],r["role"]):r for r in tracks["records"] if r["status"]=="measured"}
    rows=[]
    for eid in sorted({k[0] for k in measured}):
        if (eid,"low_response") not in measured or (eid,"high_response") not in measured: continue
        e=meta[eid]; low=measured[(eid,"low_response")]["window_median_set_max_residual_px"]; high=measured[(eid,"high_response")]["window_median_set_max_residual_px"]
        rows.append({"episode_id":eid,"failure":not e["success"],"log_response":float(np.log(e["label_blind_low_response_window"]["response_ratio"]+1e-4)),"log_low_visual":float(np.log(low+1e-4)),"visual_log_drop":float(np.log(high+1e-4)-np.log(low+1e-4))})
    labels=np.asarray([r["failure"] for r in rows],int); metrics={}
    for name,model in frozen["models"].items():
        x=np.asarray([[r[f] for f in model["feature_names"]] for r in rows]); z=(x-np.asarray(model["mean"]))/np.asarray(model["scale"])
        score=expit(model["intercept"]+z@np.asarray(model["coefficients"])); alarm=score>=model["threshold_success_q95"]
        metrics[name]={"auc":auc(score,labels),"success_alarm_rate":float(np.mean(alarm[labels==0])) if np.any(labels==0) else None,"failure_detection_rate":float(np.mean(alarm[labels==1])) if np.any(labels==1) else None,"scores":[{"episode_id":r["episode_id"],"failure":bool(y),"score":float(s),"alarm":bool(a)} for r,y,s,a in zip(rows,labels,score,alarm)]}
    payload={"schema_version":1,"evaluation_firewall":"frozen model applied without refit","episodes":len(rows),"failures":int(labels.sum()),"successes":int((1-labels).sum()),"metrics":metrics}
    args.output.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8"); print(json.dumps({k:{x:v[x] for x in ('auc','success_alarm_rate','failure_detection_rate')} for k,v in metrics.items()}))


if __name__=="__main__": main()
