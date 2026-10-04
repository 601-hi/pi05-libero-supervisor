#!/usr/bin/env python3
"""Leave-one-suite-out diagnostic for the wrist normal-response expert."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


SUITES = ("object", "goal", "libero90")


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive, negative = scores[labels], scores[~labels]
    return float(((positive[:, None] > negative).sum() + .5 * (positive[:, None] == negative).sum()) /
                 (len(positive) * len(negative)))


def fit(x: np.ndarray, y: np.ndarray, ridge: float) -> dict:
    xm, xs = x.mean(0), x.std(0); xs[xs < 1e-8] = 1
    ym = y.mean(0); xn = (x - xm) / xs
    w = np.linalg.solve(xn.T @ xn + ridge * np.eye(xn.shape[1]), xn.T @ (y - ym))
    residual = np.linalg.norm((y - (ym + xn @ w)).reshape(len(y), 2, 16), axis=1)
    center = np.median(residual, axis=0); scale = 1.4826 * np.median(np.abs(residual - center), axis=0)
    scale[scale < 1e-6] = np.maximum(residual.std(0), 1e-6)[scale < 1e-6]
    return {"xm": xm, "xs": xs, "ym": ym, "w": w, "center": center, "scale": scale}


def score(model: dict, x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    prediction = model["ym"] + ((x - model["xm"]) / model["xs"]) @ model["w"]
    actual_norm = np.linalg.norm(y, axis=1); predicted_norm = np.linalg.norm(prediction, axis=1)
    residual = np.linalg.norm((y - prediction).reshape(len(y), 2, 16), axis=1)
    patch_z = (residual - model["center"]) / model["scale"]
    return {
        "high_residual": np.partition(patch_z, -4, axis=1)[:, -4:].mean(1),
        "low_response": -np.sum(y * prediction, axis=1) / (predicted_norm ** 2 + 1e-12),
        "low_cosine": -np.sum(y * prediction, axis=1) / (actual_norm * predicted_norm + 1e-12),
        "low_raw_flow": -actual_norm,
    }


def persistent(points: np.ndarray, required: int = 2, window: int = 3) -> np.ndarray:
    result = np.zeros(len(points), dtype=bool)
    for i in range(len(points)):
        result[i] = int(points[max(0, i - window + 1):i + 1].sum()) >= required
    return result


def support_diagnostic(train_x: np.ndarray, held_x: np.ndarray) -> dict[str, float]:
    center, scale = train_x.mean(0), train_x.std(0); scale[scale < 1e-8] = 1
    train_z = (train_x - center) / scale; held_z = (held_x - center) / scale
    train_distance = np.sqrt(np.maximum(np.sum(train_z*train_z,axis=1)[:,None]+
        np.sum(train_z*train_z,axis=1)[None,:]-2*train_z@train_z.T,0))
    np.fill_diagonal(train_distance, np.inf)
    train_nearest = train_distance.min(1)
    held_distance=np.sqrt(np.maximum(np.sum(held_z*held_z,axis=1)[:,None]+
        np.sum(train_z*train_z,axis=1)[None,:]-2*held_z@train_z.T,0))
    held_nearest = held_distance.min(1)
    threshold = float(np.quantile(train_nearest, .99))
    return {"train_leave_one_out_nn_q99": threshold,
            "held_nn_median": float(np.median(held_nearest)),
            "held_nn_p90": float(np.percentile(held_nearest, 90)),
            "held_outside_train_q99_fraction": float(np.mean(held_nearest > threshold)),
            "held_max_abs_z_p90": float(np.percentile(np.max(np.abs(held_z), axis=1), 90))}


def support_mask(train_x: np.ndarray, query_x: np.ndarray) -> tuple[np.ndarray, float]:
    center, scale = train_x.mean(0), train_x.std(0); scale[scale < 1e-8] = 1
    train_z = (train_x - center) / scale; query_z = (query_x - center) / scale
    train_norm=np.sum(train_z*train_z,axis=1)
    distance = np.sqrt(np.maximum(train_norm[:,None]+train_norm[None,:]-2*train_z@train_z.T,0))
    np.fill_diagonal(distance, np.inf)
    threshold = float(np.quantile(distance.min(1), .99))
    query_distance=np.sqrt(np.maximum(np.sum(query_z*query_z,axis=1)[:,None]+train_norm[None,:]-2*query_z@train_z.T,0))
    query_nearest = query_distance.min(1)
    return query_nearest <= threshold, threshold


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--dataset',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--ridge',type=float,default=10.0)
    p.add_argument('--condition-mode',choices=('full','motion_invariant'),default='full');a=p.parse_args()
    d=np.load(a.dataset,allow_pickle=False); sid=d['sample_id'].astype(str); valid=d['one_step_valid'].astype(bool); abnormal=d['abnormal'].astype(bool)
    x=d['x_condition'].astype(float); y=d['one_step_grid'].astype(float); reports=[]
    if a.condition_mode == 'motion_invariant':
        # Keep actions, joint velocity, gripper state/velocity, chunk phase, and
        # previous realized translation. Drop absolute EEF pose and joint pose.
        columns=np.r_[0:14,28:43]; x=x[:,columns]
    for held in SUITES:
        train=valid & np.char.endswith(sid,'_normal') & ~np.char.startswith(sid,held)
        test_normal=valid & (sid==held+'_normal'); test_abnormal=valid & (sid==held+'_scale050') & abnormal
        test=test_normal|test_abnormal; labels=test_abnormal[test]
        model=fit(x[train],y[train],a.ridge); scores=score(model,x[test],y[test])
        train_scores=score(model,x[train],y[train]); thresholds={k:float(np.quantile(v,.99)) for k,v in train_scores.items()}
        normal_sequence=np.flatnonzero(test_normal); disturbed_sequence=np.flatnonzero(valid & (sid==held+'_scale050'))
        normal_supported,_=support_mask(x[train],x[normal_sequence]); disturbed_supported,_=support_mask(x[train],x[disturbed_sequence])
        operating={}
        for metric,threshold in thresholds.items():
            normal_values=score(model,x[normal_sequence],y[normal_sequence])[metric]
            disturbed_values=score(model,x[disturbed_sequence],y[disturbed_sequence])[metric]
            normal_alarm=persistent(normal_values>=threshold); disturbed_alarm=persistent(disturbed_values>=threshold)
            gated_normal_alarm=persistent((normal_values>=threshold)&normal_supported)
            gated_disturbed_alarm=persistent((disturbed_values>=threshold)&disturbed_supported)
            disturbed_active=abnormal[disturbed_sequence]
            operating[metric]={'train_normal_q99':threshold,
              'held_normal_step_alarm_rate':float(np.mean(normal_values>=threshold)),
              'held_normal_episode_alarm':bool(normal_alarm.any()),
              'active_point_recall':float(np.mean(disturbed_values[disturbed_active]>=threshold)),
              'active_persistent_recall':float(np.mean(disturbed_alarm[disturbed_active])),
              'detected_during_active':bool(np.any(disturbed_alarm&disturbed_active)),
              'support_gated':{'held_normal_coverage':float(normal_supported.mean()),
                'active_coverage':float(disturbed_supported[disturbed_active].mean()),
                'held_normal_step_alarm_rate':float(np.mean((normal_values>=threshold)&normal_supported)),
                'held_normal_episode_alarm':bool(gated_normal_alarm.any()),
                'active_persistent_recall':float(np.mean(gated_disturbed_alarm[disturbed_active])),
                'detected_during_active':bool(np.any(gated_disturbed_alarm&disturbed_active))}}
        reports.append({'held_suite':held,'train_normal_rows':int(train.sum()),'test_normal_rows':int(test_normal.sum()),
                        'test_active_rows':int(test_abnormal.sum()),'auc':{k:auc(labels,v) for k,v in scores.items()},
                        'condition_support':support_diagnostic(x[train],x[test_normal]),
                        'q99_operating_point':operating})
    report={'status':'development pilot; one seed and one pair per suite','task_id_used':False,'anomaly_labels_used_for_fit':False,
            'condition_mode':a.condition_mode,'condition_dim':int(x.shape[1]),'ridge':a.ridge,'reports':reports}
    a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
