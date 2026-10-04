"""Shrunk task-conditional visual normal/abnormal density experts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


LOG2PI = float(np.log(2 * np.pi))


def robust_scale(values):
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) * 1.4826
    if mad < 1e-6:
        mad = float(np.std(values))
    return median, max(mad, 1e-3)


def fit(data, feature_matrix, pca_dim, shrinkage):
    ambiguous = data["previous_ambiguous_any"].astype(bool)
    active = data["previous_active_any"].astype(bool)
    clean = np.isclose(data["scale"], 1.0)
    selected = ambiguous & (clean | active)
    x = feature_matrix[:, :256].astype(np.float64)
    mean, scale = x[selected].mean(0), x[selected].std(0)
    scale[scale < 1e-6] = 1.0
    standardized = (x - mean) / scale
    _, _, vt = np.linalg.svd(standardized[selected], full_matrices=False)
    projection = vt[:pca_dim].T
    z = standardized @ projection
    label = active
    global_stats = {}
    for cls in (0, 1):
        values = z[selected & (label == bool(cls))]
        global_stats[cls] = (values.mean(0), values.var(0) + 1e-3)
    task_mean = np.zeros((10, 2, pca_dim)); task_var = np.zeros_like(task_mean); task_count = np.zeros((10, 2), int)
    for task in range(10):
        for cls in (0, 1):
            values = z[selected & (data["task_id"] == task) & (label == bool(cls))]
            gm, gv = global_stats[cls]
            n = len(values); task_count[task, cls] = n; weight = n / (n + shrinkage)
            tm = values.mean(0) if n else gm
            tv = values.var(0) + 1e-3 if n else gv
            combined_mean = weight * tm + (1-weight) * gm
            combined_var = weight * (tv + (tm-combined_mean)**2) + (1-weight) * (gv + (gm-combined_mean)**2)
            task_mean[task, cls], task_var[task, cls] = combined_mean, np.maximum(combined_var, 1e-3)
    model = {"mean": mean, "scale": scale, "projection": projection, "task_mean": task_mean,
             "task_var": task_var, "task_count": task_count, "pca_dim": pca_dim, "shrinkage": shrinkage}
    raw = score_raw(model, data, feature_matrix)
    centers=np.zeros(10); scales=np.ones(10)
    for task in range(10):
        values=raw[ambiguous & clean & (data["task_id"]==task)]
        centers[task],scales[task]=robust_scale(values)
    model["normal_center"],model["normal_scale"]=centers,scales
    return model


def score_raw(model, data, feature_matrix):
    x=feature_matrix[:,:256].astype(np.float64)
    z=((x-model["mean"])/model["scale"])@model["projection"]
    task=data["task_id"].astype(int); logp=[]
    for cls in (0,1):
        m=model["task_mean"][task,cls];v=model["task_var"][task,cls]
        logp.append(-.5*np.sum(np.log(v)+(z-m)**2/v+LOG2PI,axis=1))
    return logp[1]-logp[0]


def score(model, data, feature_matrix):
    raw=score_raw(model,data,feature_matrix);task=data["task_id"].astype(int)
    return (raw-model["normal_center"][task])/model["normal_scale"][task]


def save_model(path: Path, model):
    np.savez_compressed(path, **{key: np.asarray(value) for key,value in model.items()})


def load_model(path: Path):
    with np.load(path,allow_pickle=False) as d:return {key:d[key] for key in d.files}
