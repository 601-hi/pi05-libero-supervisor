#!/usr/bin/env python3
"""Fit median selective gates and reliability references from generic source scores."""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--scores',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--tune-output',type=Path,required=True);a=p.parse_args();d=np.load(a.scores);fit=d['reliability_fit'].astype(bool);normal=~d['abnormal'];abnormal=d['abnormal'];lr=d['abnormal_logp']-d['normal_logp'];best=np.maximum(d['normal_logp'],d['abnormal_logp'])
 nf=fit&normal;af=fit&abnormal;tn=(~fit)&normal;ta=(~fit)&abnormal;normal_gate=float(np.median(lr[nf]));abnormal_gate=float(np.median(lr[af]));payload={'schema_version':2,'source_only':True,'score_sha256':sha(a.scores),'best_logp_reference':best[fit].tolist(),'ensemble_std_reference':d['ensemble_std'][fit].tolist(),'normal_lr_threshold':normal_gate,'abnormal_lr_threshold':abnormal_gate,'overlap_warning':normal_gate>=abnormal_gate,'split_rule':'episode parity; even reliability fit, odd sequence tune','counts':{'fit_normal':int(nf.sum()),'fit_abnormal':int(af.sum()),'tune_normal':int(tn.sum()),'tune_abnormal':int(ta.sum())}}
 a.output.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8');keys=('normal_logp','abnormal_logp','ensemble_std','abnormal','scale','episode','action');tune=~fit;np.savez_compressed(a.tune_output,**{k:d[k][tune] for k in keys});print(json.dumps({k:payload[k] for k in ('normal_lr_threshold','abnormal_lr_threshold','overlap_warning','counts')},indent=2))
if __name__=='__main__':main()
