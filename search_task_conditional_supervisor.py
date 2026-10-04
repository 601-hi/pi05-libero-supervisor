#!/usr/bin/env python3
"""Search task-conditional conformal residual rules on development splits."""

import numpy as np

import evaluate_supervisor as ev
import normal_mechanism_analysis as na

BASE = "/root/gpufree-data/libero-traces/normal_kinematics_spatial_5ep_seed7_noise20260901.jsonl"
NORMAL = "/root/gpufree-data/libero-traces/normal_independent_spatial_10ep_seed11_noise20260902.jsonl"
ANOM = [
    "/root/gpufree-data/libero-traces/independent_disturb_scale025_start40_len10_spatial_3ep_seed11_noise20260902.jsonl",
    "/root/gpufree-data/libero-traces/independent_disturb_scale050_start40_len10_spatial_3ep_seed11_noise20260902.jsonl",
    "/root/gpufree-data/libero-traces/independent_disturb_scale075_start40_len10_spatial_3ep_seed11_noise20260902.jsonl",
]

def load(path):
    rows = na.load_jsonl(path)
    return rows, na.build_dataset(ev.force_include_all_episodes(rows))

def score_dataset(model, covariance, path):
    rows, (records, _, x, y, _) = load(path)
    pred = na.predict(model, x)
    score = ev.directional_shortfall(pred, y, pred, covariance)
    active_lookup = {(r["task_id"], r["episode_idx"], r["action_index"]): bool(r.get("disturbance_active"))
                     for r in rows if r.get("event") == "step"}
    active = np.asarray([active_lookup.get((r["task_id"], r["episode_idx"], r["action_index"]), False)
                         for r in records])
    for r, flag in zip(records, active): r["disturbance_active"] = bool(flag)
    return records, score, active

def episode_detection(records, alarm, active):
    detected = 0
    for indices in ev.episode_groups(records).values():
        if any(active[i] for i in indices) and any(alarm[i] and active[i] for i in indices): detected += 1
    return detected

def main():
    _, (br, _, bx, by, _) = load(BASE)
    _, (nr, _, nx, ny, _) = load(NORMAL)
    ep = np.asarray([r["episode_idx"] for r in nr])
    dev, cal, test = ep <= 5, (ep >= 6) & (ep <= 7), ep >= 8
    model = na.fit_ridge(np.r_[bx, nx[dev]], np.r_[by, ny[dev]], nx[cal], ny[cal])
    cal_pred = na.predict(model, nx[cal])
    covariance, _ = ev.covariance_inverse(ny[cal] - cal_pred)
    cal_score = ev.directional_shortfall(cal_pred, ny[cal], cal_pred, covariance)
    cal_records = [r for r, k in zip(nr, cal) if k]
    test_pred = na.predict(model, nx[test])
    test_score = ev.directional_shortfall(test_pred, ny[test], test_pred, covariance)
    test_records = [r for r, k in zip(nr, test) if k]
    anomalies = [score_dataset(model, covariance, p) for p in ANOM]
    results = []
    for q in (95, 97, 97.5, 98, 98.5, 99, 99.25, 99.5, 99.75):
        thresholds = {task: float(np.percentile([s for s, r in zip(cal_score, cal_records) if r["task_id"] == task], q))
                      for task in range(10)}
        for min_target in (0.01, 0.015, 0.02, 0.025):
          for max_progress in (0.10, 0.15, 0.20, 0.30):
            for required, window in ((1,1),(2,3),(3,5),(4,7)):
                def alarms(records, scores):
                    point = np.asarray([s > thresholds[r["task_id"]] and r["target_norm"] >= min_target
                                        and r["progress"] < max_progress for r, s in zip(records, scores)])
                    return ev.persistence_series(point.astype(float), records, .5, required=required, window=window)
                normal_alarm = alarms(test_records, test_score)
                normal_eps = ev.episode_alarm_count(test_records, normal_alarm, .5)
                if normal_eps > 2: continue
                detections=[]; outside=[]; recalls=[]
                for records, scores, active in anomalies:
                    alarm=alarms(records,scores)
                    detections.append(episode_detection(records,alarm,active))
                    outside.append(float(alarm[~active].mean()))
                    recalls.append(float(alarm[active].mean()))
                utility=2*detections[0]+2*detections[1]+detections[2]-8*normal_eps
                results.append((utility, normal_eps, detections, recalls, outside, q, min_target,max_progress,required,window,thresholds))
    results.sort(key=lambda x:(-x[0],x[1],-sum(x[2])))
    print("SPLITS",len(br),int(dev.sum()),int(cal.sum()),int(test.sum()),"NORMAL_TEST_EPISODES",len(ev.episode_groups(test_records)))
    for row in results[:30]:
        utility, normal_eps, detections, recalls, outside, q, min_target,max_progress,required,window,thresholds=row
        print({"utility":utility,"normal_alarm_episodes":normal_eps,"detections_of_30":detections,
               "active_recalls":recalls,"outside_rates":outside,"q":q,"min_target":min_target,
               "max_progress":max_progress,"persistence":f"{required}of{window}",
               "threshold_range":[min(thresholds.values()),max(thresholds.values())]})

if __name__ == "__main__": main()
