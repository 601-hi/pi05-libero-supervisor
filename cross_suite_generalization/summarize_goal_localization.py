"""Candidate coverage (not accuracy) and outcome-free review sheets."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from PIL import Image, ImageDraw


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    args=p.parse_args()
    rows=[json.loads(line) for line in (args.run/'predictions.jsonl').open(encoding='utf-8')]
    coverage=defaultdict(Counter)
    for r in rows:
        for role,query in r['roles']:
            if role == 'destination' and r['task_language'].startswith(('open ','close ')):
                role='context'
            matches=[b for b in r['boxes'] if b['label'].strip().lower() == query.lower()]
            count=coverage[r['camera']+':'+role]
            count['frames']+=1
            count['has_exact_label_candidate']+=bool(matches)
            count['multiple_candidates']+=len(matches)>1
    for camera in ('agent_images','wrist_images'):
        selected=[r for r in rows if r['camera']==camera and r['action_index']==40]
        for start in range(0,len(selected),10):
            part=selected[start:start+10]
            canvas=Image.new('RGB',(5*320,2*360),'white')
            draw=ImageDraw.Draw(canvas)
            for slot,r in enumerate(part):
                x=(slot%5)*320;y=(slot//5)*360
                draw.text((x+3,y+2),f'{start+slot} {r["id"]}',fill='black')
                draw.text((x+3,y+16),r['task_language'][:48],fill='black')
                canvas.paste(Image.open(args.run/(r['id']+'.jpg')).resize((320,320)),(x,y+36))
            canvas.save(args.run/f'review_{camera}_{start:02d}.jpg',quality=95)
    payload=dict(status='candidate_coverage_is_not_accuracy',coverage=dict(coverage),images=len(rows))
    (args.run/'coverage.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps(payload))


if __name__=='__main__': main()
