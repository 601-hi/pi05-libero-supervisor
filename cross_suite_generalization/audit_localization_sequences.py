"""Audit proposal ambiguity and instability; no localization truth is assumed."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--predictions',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    groups=defaultdict(list)
    episodes=set();frames=0
    with args.predictions.open(encoding='utf-8') as handle:
        for line in handle:
            row=json.loads(line);frames+=1;episodes.add(row['episode_key'])
            for role,query in row['roles']:
                if role == 'destination' and row['task_language'].startswith(('open ','close ')):
                    role='context'  # Normalize legacy pilot metadata; prompts were unchanged.
                boxes=[b for b in row['boxes'] if b['label'].strip().lower()==query.lower()]
                top=max(boxes,key=lambda b:b['score'] if b['score'] is not None else 0) if boxes else None
                groups[row['camera']+':'+role].append(dict(episode=row['episode_key'],action=row['action_index'],
                    count=len(boxes),box=top['box_xyxy'] if top else None,width=row['width'],height=row['height']))
    reports={}
    for key,rows in groups.items():
        prior={};areas=[];top_border=0;jumps=[];valid_pairs=0
        for row in rows:
            box=row['box'];previous=prior.get(row['episode'])
            if box is not None:
                x1,y1,x2,y2=box
                areas.append((x2-x1)*(y2-y1)/(row['width']*row['height']))
                top_border+=y1<=1
                if previous is not None and previous['box'] is not None and row['action']-previous['action']==10:
                    a=np.asarray(box);b=np.asarray(previous['box'])
                    center_a=(a[:2]+a[2:])/2;center_b=(b[:2]+b[2:])/2
                    jumps.append(float(np.linalg.norm((center_a-center_b)/[row['width'],row['height']])))
                    valid_pairs+=1
            prior[row['episode']]=row
        reports[key]=dict(frames=len(rows),missing=sum(r['count']==0 for r in rows),
            multiple=sum(r['count']>1 for r in rows),detected=len(areas),
            top_proposal_touches_image_top=top_border,
            top_box_area_fraction_q10_q50_q90=np.quantile(areas,[.1,.5,.9]).tolist() if areas else [],
            adjacent_top_proposal_pairs=valid_pairs,
            top_center_jump_over_015_image_units=sum(x>.15 for x in jumps),
            top_center_jump_q50_q90=np.quantile(jumps,[.5,.9]).tolist() if jumps else [])
    payload=dict(episodes=len(episodes),images=frames,reports=reports,
        warning='Top-score boxes (first-returned if scores absent) are diagnostic only, not selected physical targets. Border touch and jumps are not error labels. No precision/recall can be computed without location truth.')
    args.output.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps(payload,indent=2))


if __name__=='__main__':main()
