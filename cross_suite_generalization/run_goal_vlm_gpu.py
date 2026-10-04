"""Fixed seed36 RGB-only pilot; semantic outputs require independent review."""
import argparse
from collections import defaultdict,Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image,ImageDraw
import torch
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from vla_supervisor.goal_vlm_output import parse_goal_vlm

PROMPT = '''Analyze two current robot images. Image 1 is the fixed external camera. Image 2 is the wrist camera. Task: {task}
Return exactly one JSON object, without markdown, with fields:
"gripper_center": [x,y] for the point BETWEEN THE TWO FINGERTIPS in image 1, NOT the robot arm or wrist body, or null if uncertain;
"object_box": [x1,y1,x2,y2] for the task's manipulated object or operational part in image 1, or null if hidden or identity ambiguous;
"destination_box": [x1,y1,x2,y2] for the placement destination in image 1, or null if no placement or unclear;
"phase": one of approach, grasp, transport, place, manipulate, retract, unknown;
"goal_state": achieved, not_achieved, or unknown;
"evidence": one short sentence describing only visible evidence and uncertainties.
Coordinates use 0 to 1000 relative to image 1. Do not output coordinates from image 2. Do not assume an object is grasped just because fingers are close. A single instant cannot establish progress or failure. Use null/unknown instead of inventing occluded locations. Ground objects by the complete task description, not only color.'''


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input-dir',type=Path,required=True)
    p.add_argument('--model-info',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--action-index',type=int,default=40)
    p.add_argument('--negative-controls',action='store_true')
    args=p.parse_args()
    info=json.loads(args.model_info.read_text())
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if (args.output_dir/'predictions.jsonl').exists():
        raise FileExistsError('new run requires a new output directory')
    torch.set_num_threads(2)
    model=Qwen3VLForConditionalGeneration.from_pretrained(info['snapshot'],local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    processor=AutoProcessor.from_pretrained(info['snapshot'],local_files_only=True)
    lookup=json.loads((args.input_dir/'io_lookup_not_model_input.json').read_text())
    jobs={}
    with (args.input_dir/'causal_inputs.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            row=json.loads(line)
            if row['action_index']==args.action_index:
                jobs.setdefault(row['task_language'],row)
    counts=Counter(); start=time.monotonic()
    with (args.output_dir/'predictions.jsonl').open('x',encoding='utf-8') as output:
        for n,row in enumerate(jobs.values()):
            with np.load(lookup[row['episode_key']]['sidecar_path'],allow_pickle=False) as archive:
                idx=row['image_frame_index']
                if int(archive['action_indices'][idx]) != row['action_index']:
                    raise ValueError('frame/action mismatch')
                images=[Image.fromarray(archive[c][idx]) for c in ('agent_images','wrist_images')]
            prompt_task=('pick up the elephant and place it on the traffic light'
                         if args.negative_controls else row['task_language'])
            prompt=PROMPT.format(task=prompt_task)
            messages=[{'role':'user','content':[{'type':'image','image':im} for im in images]+[{'type':'text','text':prompt}]}]
            inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                return_dict=True,return_tensors='pt').to('cuda')
            with torch.inference_mode():
                generated=model.generate(**inputs,max_new_tokens=300,do_sample=False)
            text=processor.batch_decode(generated[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)[0]
            parsed=parse_goal_vlm(text)
            counts['valid' if parsed['valid'] else 'invalid']+=1
            opaque=hashlib.sha256(f'{row["episode_key"]}:{row["action_index"]}:agent_images'.encode()).hexdigest()[:16]
            record=dict(id=opaque,episode_key=row['episode_key'],action_index=row['action_index'],
                task_language=row['task_language'],prompt_task=prompt_task,
                negative_control=args.negative_controls,raw_output=text,parsed=parsed,
                inference_tokens=int(generated.shape[1]-inputs['input_ids'].shape[1]))
            output.write(json.dumps(record,ensure_ascii=False)+'\n');output.flush()
            canvas=images[0].resize((448,448));draw=ImageDraw.Draw(canvas)
            if parsed['valid']:
                for key,color in [('object_box','red'),('destination_box','cyan')]:
                    if parsed[key] is not None:
                        draw.rectangle([v*.448 for v in parsed[key]],outline=color,width=2)
                if parsed['gripper_center'] is not None:
                    x,y=[v*.448 for v in parsed['gripper_center']]
                    draw.ellipse((x-5,y-5,x+5,y+5),outline='lime',width=3)
            canvas.save(args.output_dir/(opaque+'.jpg'),quality=95)
            print(json.dumps(dict(completed=n+1,elapsed_s=round(time.monotonic()-start,1),valid=parsed['valid'])),flush=True)
    (args.output_dir/'summary.json').write_text(json.dumps(dict(model=info['repo'],revision=info['revision'],
        prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),tasks=len(jobs),counts=counts,
        elapsed_s=time.monotonic()-start,negative_controls=args.negative_controls,
        action_index=args.action_index,calibrated=False),indent=2),encoding='utf-8')


if __name__=='__main__': main()
