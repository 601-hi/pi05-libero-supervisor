#!/usr/bin/env python3
"""Train dual conditional density experts on natural success/failure responses.

This is an outcome-risk model, not a hardware-fault oracle.  Success/failure is
weak episode supervision.  Every episode contributes at most ``samples_per_episode``
uniformly spaced steps, preventing timeout failures from dominating by length.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def suite_from_path(path: Path) -> str:
    text = str(path).lower()
    for name in ("libero_90", "libero_10", "libero_goal", "libero_object", "libero_spatial"):
        if name in text:
            return name
    return "unknown"


def quaternion_delta_rotvec(after, before):
    """Shortest relative quaternion as a small rotation vector (xyzw input)."""
    # LIBERO observations are xyzw. q_rel = q_after * conjugate(q_before).
    a = np.asarray(after, float); b = np.asarray(before, float)
    av, aw = a[:3], a[3]; bv, bw = -b[:3], b[3]
    vector = aw * bv + bw * av + np.cross(av, bv)
    scalar = aw * bw - float(av @ bv)
    if scalar < 0:
        vector, scalar = -vector, -scalar
    norm = np.linalg.norm(vector)
    if norm < 1e-10:
        return 2.0 * vector
    return vector / norm * (2.0 * math.atan2(norm, max(scalar, 0.0)))


def load_episodes(root: Path):
    episodes = []
    for path in sorted((root / "traces").glob("*.jsonl")):
        grouped, endings = defaultdict(list), {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("event") == "step":
                    grouped[(int(row["task_id"]), int(row["episode_idx"]))].append(row)
                elif row.get("event") == "episode_end":
                    endings[(int(row["task_id"]), int(row["episode_idx"]))] = row
        for key, rows in grouped.items():
            if key not in endings or not rows:
                continue
            rows.sort(key=lambda row: int(row["action_index"]))
            if any(bool(row.get("disturbance_active", False)) for row in rows):
                continue
            suite = suite_from_path(path)
            episodes.append({"id": f"{suite}:task{key[0]}:episode{key[1]}", "suite": suite,
                             "task_id": key[0], "episode_idx": key[1],
                             "success": bool(endings[key].get("success", False)), "rows": rows})
    return episodes


def episode_arrays(episode, samples_per_episode):
    rows = episode["rows"]
    indices = np.unique(np.linspace(0, len(rows) - 1, min(samples_per_episode, len(rows))).round().astype(int))
    initial_eef = np.asarray(rows[0]["eef_pos_before"], float)
    x, y, actions = [], [], []
    previous_response = np.zeros(22, float)
    for i, row in enumerate(rows):
        q0 = np.asarray(row["joint_pos_before"], float)
        q1 = np.asarray(row["joint_pos_after"], float)
        v0 = np.asarray(row["joint_vel_before"], float)
        v1 = np.asarray(row["joint_vel_after"], float)
        p0 = np.asarray(row["eef_pos_before"], float)
        p1 = np.asarray(row["eef_pos_after"], float)
        g0 = np.asarray(row["gripper_qpos_before"], float)
        g1 = np.asarray(row["gripper_qpos_after"], float)
        response = np.r_[q1 - q0, v1 - v0, p1 - p0,
                         quaternion_delta_rotvec(row["eef_quat_after"], row["eef_quat_before"]), g1 - g0]
        if i in indices:
            action = np.asarray(row.get("intended_action", row["action"]), float)
            context = np.r_[action, q0, v0, p0 - initial_eef,
                            np.asarray(row["eef_quat_before"], float), g0, previous_response]
            x.append(context); y.append(response); actions.append(int(row["action_index"]))
        previous_response = response
    return np.asarray(x, np.float32), np.asarray(y, np.float32), np.asarray(actions, np.int32)


def robust_fit(values):
    center = np.median(values, axis=0)
    scale = 1.4826 * np.median(np.abs(values - center), axis=0)
    scale = np.maximum(scale, np.std(values, axis=0) * 0.1)
    return center.astype(np.float32), np.maximum(scale, 1e-6).astype(np.float32)


def auc(labels, scores):
    labels, scores = np.asarray(labels, bool), np.asarray(scores, float)
    pos, neg = scores[labels], scores[~labels]
    if not len(pos) or not len(neg): return None
    return float((pos[:, None] > neg).mean() + 0.5 * (pos[:, None] == neg).mean())


def make_model(torch, nn, input_dim, output_dim, seed):
    torch.manual_seed(seed)
    class DualExpert(nn.Module):
        def __init__(self):
            super().__init__()
            def expert():
                return nn.Sequential(nn.Linear(input_dim, 96), nn.SiLU(), nn.Linear(96, 96),
                                     nn.SiLU(), nn.Linear(96, 2 * output_dim))
            self.normal, self.abnormal = expert(), expert()
        def logp(self, x, y, abnormal=False):
            raw = (self.abnormal if abnormal else self.normal)(x)
            mean, logvar = raw[:, :output_dim], raw[:, output_dim:].clamp(-7, 5)
            return -.5 * (logvar + (y - mean).square() * torch.exp(-logvar) + np.log(2*np.pi)).sum(1)
    return DualExpert()


def train_model(episodes, samples, args, seed):
    import torch
    from torch import nn
    normal = [samples[e["id"]] for e in episodes if e["success"]]
    abnormal = [samples[e["id"]] for e in episodes if not e["success"]]
    if not normal or not abnormal: raise ValueError("both success and failure episodes are required")
    nx, ny = np.concatenate([z[0] for z in normal]), np.concatenate([z[1] for z in normal])
    ax, ay = np.concatenate([z[0] for z in abnormal]), np.concatenate([z[1] for z in abnormal])
    xcenter, xscale = robust_fit(np.concatenate([nx, ax])); ycenter, yscale = robust_fit(np.concatenate([ny, ay]))
    norm = lambda x, c, s: (x-c)/s
    nx, ax = norm(nx,xcenter,xscale), norm(ax,xcenter,xscale)
    ny, ay = norm(ny,ycenter,yscale), norm(ay,ycenter,yscale)
    model = make_model(torch, nn, nx.shape[1], ny.shape[1], seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    tensors = [torch.from_numpy(z.astype(np.float32)) for z in (nx,ny,ax,ay)]
    NX,NY,AX,AY=tensors; rng=torch.Generator().manual_seed(seed); history=[]
    steps=max(math.ceil(len(NX)/args.batch_size), math.ceil(len(AX)/args.batch_size))
    for epoch in range(args.epochs):
        model.train(); losses=[]
        ni=torch.randint(len(NX),(steps*args.batch_size,),generator=rng)
        ai=torch.randint(len(AX),(steps*args.batch_size,),generator=rng)
        for start in range(0,len(ni),args.batch_size):
            xn,yn=NX[ni[start:start+args.batch_size]],NY[ni[start:start+args.batch_size]]
            xa,ya=AX[ai[start:start+args.batch_size]],AY[ai[start:start+args.batch_size]]
            nnlp=model.logp(xn,yn,False); nalp=model.logp(xn,yn,True)
            aalp=model.logp(xa,ya,True); anlp=model.logp(xa,ya,False)
            loss=-.5*(nnlp.mean()+aalp.mean())+args.margin_weight*(
                torch.relu(args.margin-(nnlp-nalp)).mean()+torch.relu(args.margin-(aalp-anlp)).mean())
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);opt.step();losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    return model,{"xcenter":xcenter,"xscale":xscale,"ycenter":ycenter,"yscale":yscale,
                  "history":history,"normal_rows":len(nx),"abnormal_rows":len(ax)}


def score_episodes(model, stats, episodes, samples):
    import torch
    rows=[];model.eval()
    with torch.no_grad():
        for ep in episodes:
            x,y,actions=samples[ep["id"]]
            X=torch.from_numpy(((x-stats["xcenter"])/stats["xscale"]).astype(np.float32))
            Y=torch.from_numpy(((y-stats["ycenter"])/stats["yscale"]).astype(np.float32))
            score=(model.logp(X,Y,True)-model.logp(X,Y,False)).numpy()
            rows.append({"id":ep["id"],"suite":ep["suite"],"task_id":ep["task_id"],"episode_idx":ep["episode_idx"],
                         "success":ep["success"],"steps":len(ep["rows"]),"sampled_steps":len(score),
                         "score_mean":float(np.mean(score)),"score_q90":float(np.quantile(score,.9)),
                         "score_max":float(np.max(score)),"positive_fraction":float(np.mean(score>0))})
    return rows


def summarize(rows, threshold=None):
    labels=[not x["success"] for x in rows]
    out={"episodes":len(rows),"successes":sum(x["success"] for x in rows),"failures":sum(labels)}
    for key in ("score_mean","score_q90","score_max","positive_fraction"):
        out[f"{key}_auc"]=auc(labels,[x[key] for x in rows])
    if threshold is not None:
        for x in rows:x["alarm_q90"]=x["score_q90"]>threshold
        ok=[x for x in rows if x["success"]];bad=[x for x in rows if not x["success"]]
        out["q90_threshold"]=threshold
        out["success_alarm_rate"]=float(np.mean([x["alarm_q90"] for x in ok])) if ok else None
        out["failure_detection_rate"]=float(np.mean([x["alarm_q90"] for x in bad])) if bad else None
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--output-directory",type=Path,required=True)
    p.add_argument("--samples-per-episode",type=int,default=96);p.add_argument("--epochs",type=int,default=60);p.add_argument("--batch-size",type=int,default=256)
    p.add_argument("--learning-rate",type=float,default=1e-3);p.add_argument("--margin",type=float,default=2.0);p.add_argument("--margin-weight",type=float,default=.5);p.add_argument("--seed",type=int,default=202609111);p.add_argument("--threads",type=int,default=2);args=p.parse_args()
    import torch;torch.set_num_threads(args.threads)
    episodes=load_episodes(args.root);samples={e["id"]:episode_arrays(e,args.samples_per_episode) for e in episodes}
    failure_groups=sorted({(e["suite"],e["task_id"]) for e in episodes if not e["success"]})
    folds=[]
    for fold_index,group in enumerate(failure_groups):
        train=[e for e in episodes if (e["suite"],e["task_id"])!=group]
        test=[e for e in episodes if (e["suite"],e["task_id"])==group]
        model,stats=train_model(train,samples,args,args.seed+fold_index)
        train_rows=score_episodes(model,stats,train,samples);test_rows=score_episodes(model,stats,test,samples)
        train_success_q90=[x["score_q90"] for x in train_rows if x["success"]]
        threshold=float(np.quantile(train_success_q90,.90,method="higher"))
        folds.append({"held_out_group":{"suite":group[0],"task_id":group[1]},
                      "train":summarize(train_rows),"test":summarize(test_rows,threshold),"details":test_rows})
        print(json.dumps({"held_out_group":folds[-1]["held_out_group"],"test":folds[-1]["test"]},ensure_ascii=False))
    final_model,final_stats=train_model(episodes,samples,args,args.seed+100)
    args.output_directory.mkdir(parents=True,exist_ok=True)
    checkpoint={"state_dict":final_model.state_dict(),**{k:v for k,v in final_stats.items() if k!="history"},
                "config":{"input_dim":int(next(iter(samples.values()))[0].shape[1]),"output_dim":22,"seed":args.seed+100,
                          "samples_per_episode":args.samples_per_episode,"epochs":args.epochs,"margin":args.margin,
                          "margin_weight":args.margin_weight},"feature_definition":{"response":"dq7,dv7,dx3,dtheta3,dgripper2",
                          "context":"action7,q7,qvel7,eef_relative3,eef_quat4,gripper2,previous_response22"}}
    torch.save(checkpoint,args.output_directory/"natural_multiresponse_dual_expert.pt")
    report={"label_semantics":"weak episode-level natural outcome risk; not hardware fault",
            "target_used":"22D joint/eef/gripper response","episode_equalized_sampling":True,
            "episodes":len(episodes),"successes":sum(e["success"] for e in episodes),"failures":sum(not e["success"] for e in episodes),
            "failure_groups":[{"suite":x[0],"task_id":x[1]} for x in failure_groups],"folds":folds,
            "final":{"normal_rows":final_stats["normal_rows"],"abnormal_rows":final_stats["abnormal_rows"],
                     "final_loss":final_stats["history"][-1],"checkpoint":"natural_multiresponse_dual_expert.pt"}}
    (args.output_directory/"natural_multiresponse_dual_expert_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (args.output_directory/"natural_multiresponse_dual_expert_history.json").write_text(json.dumps(final_stats["history"],indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="folds"},ensure_ascii=False,indent=2))

if __name__=="__main__":main()
