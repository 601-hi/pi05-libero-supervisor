"""Create review sheets, without treating model outputs as ground truth."""
import json
import sys
from pathlib import Path
from collections import Counter
from PIL import Image, ImageDraw

root = Path(sys.argv[1])
rows = [json.loads(s) for s in (root / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
for start in range(0, len(rows), 10):
    sheet = Image.new('RGB', (1600, 1160), 'white')
    draw = ImageDraw.Draw(sheet)
    for j, row in enumerate(rows[start:start+10]):
        x, y = (j % 5)*320, (j // 5)*580
        sheet.paste(Image.open(root / (row['id']+'.jpg')).resize((320,320)), (x,y))
        label = str(start+j)+': '+row['task_language']
        lines = [label[k:k+42] for k in range(0,len(label),42)]
        lines += [str(row['parsed'].get('phase', row['parsed'].get('reason')))]
        draw.multiline_text((x+3,y+325), '\n'.join(lines), fill='black')
    sheet.save(root / ('review_%02d.jpg' % start))
valid = [r['parsed'] for r in rows if r['parsed']['valid']]
summary = {'count':len(rows), 'valid':len(valid), 'phase':dict(Counter(r['phase'] for r in valid)),
           'nonnull':{k:sum(r[k] is not None for r in valid) for k in ('object_box','destination_box','gripper_center')},
           'accuracy':None, 'reason':'No independent reference labels'}
(root/'review_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary))
