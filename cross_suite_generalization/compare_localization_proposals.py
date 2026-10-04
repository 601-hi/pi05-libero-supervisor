"""Compare two proposal sets without treating consensus as location truth."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    area_a=max(0,a[2]-a[0])*max(0,a[3]-a[1])
    area_b=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/(area_a+area_b-inter) if area_a+area_b-inter>0 else 0


def load(path):
    with path.open(encoding='utf-8') as handle:
        return {(r['episode_key'],r['action_index'],r['camera']):r
                for line in handle if (r:=json.loads(line))['camera']=='agent_images'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--a',type=Path,required=True);p.add_argument('--b',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    a,b=load(args.a),load(args.b);results=defaultdict(lambda:defaultdict(int))
    for key in a.keys()&b.keys():
        for role,query in a[key]['roles']:
            if role=='destination' and a[key]['task_language'].startswith(('open ','close ')):role='context'
            aa=[x for x in a[key]['boxes'] if x['label'].lower()==query.lower()]
            bb=[x for x in b[key]['boxes'] if x['label'].lower()==query.lower()]
            stats=results[role];stats['queries']+=1
            stats['both_have_candidates']+=bool(aa and bb)
            best=max((iou(x['box_xyxy'],y['box_xyxy']) for x in aa for y in bb),default=0)
            stats['any_pair_iou_ge_050']+=best>=.5
            stats['both_unique_and_iou_ge_050']+=len(aa)==len(bb)==1 and best>=.5
    result=dict(paired_images=len(a.keys()&b.keys()),roles={k:dict(v) for k,v in results.items()},
        warning='Agreement is not correctness; both detectors may share the same wrong arm/background target. No acceptance gate is calibrated here.')
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
