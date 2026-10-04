"""Development RGB grounding with opaque output IDs and an outcome firewall.

Pilot selection: first episode per language, fixed actions 0/40/80. The model
receives image pixels and noun queries only, never lookup filenames or labels.
Detection scores are uncalibrated. This script produces measurements, not alarms.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
import hashlib

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForZeroShotObjectDetection, AutoModelForCausalLM, AutoProcessor
from vla_supervisor.goal_relations import visual_entity_query


def queries(row):
    goal = row['goal']
    pairs = [('object', visual_entity_query(goal['manipulated_object'])), ('gripper', 'robot gripper')]
    if goal.get('target'):
        role='destination' if goal['relation'] in ('place_in','place_on') else 'context'
        pairs.append((role, visual_entity_query(goal['target'])))
    # Preserve multiple roles if their surface query coincides.
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--model', default='IDEA-Research/grounding-dino-base')
    p.add_argument('--backend', choices=('dino','florence'), default='dino')
    p.add_argument('--action-index', type=int)
    p.add_argument('--camera', choices=('both','agent_images','wrist_images'), default='both')
    p.add_argument('--negative-controls', action='store_true')
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required; do not silently consume CPU hours')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir/'predictions.jsonl'
    if predictions_path.exists():
        raise FileExistsError('Refusing to overwrite an existing inference run')
    lookup = json.loads((args.input_dir/'io_lookup_not_model_input.json').read_text(encoding='utf-8'))
    grouped = defaultdict(list)
    first = {}
    with (args.input_dir/'causal_inputs.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            if not row['visual_query_due']:
                continue
            if args.action_index is not None and row['action_index'] != args.action_index:
                continue
            first.setdefault(row['task_language'], row['episode_key'])
            if args.pilot and (row['episode_key'] != first[row['task_language']] or row['action_index'] not in (0,40,80)):
                continue
            grouped[row['episode_key']].append(row)
    torch.set_num_threads(2)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, trust_remote_code=args.backend=='florence')
    if args.backend == 'dino':
        model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model, local_files_only=True).to('cuda').eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
            trust_remote_code=True, dtype=torch.float16, attn_implementation='eager').to('cuda').eval()
    total, start = 0, time.monotonic()
    with predictions_path.open('x', encoding='utf-8') as output:
        for key, rows in grouped.items():
            with np.load(lookup[key]['sidecar_path'], allow_pickle=False) as archive:
                indices = archive['action_indices']
                for camera in ('agent_images', 'wrist_images'):
                    if args.camera != 'both' and args.camera != camera:
                        continue
                    images = archive[camera]
                    for row in rows:
                        index = row['image_frame_index']
                        if int(indices[index]) != row['action_index']:
                            raise ValueError('frame/action alignment failure')
                        image = Image.fromarray(images[index])
                        roles = queries(row)
                        if args.negative_controls:
                            roles=[('absent_control_1','elephant'),('absent_control_2','traffic light')]
                        caption = '. '.join(dict.fromkeys(q for _,q in roles)) + '.'
                        if args.backend == 'dino':
                            inputs = processor(images=image, text=caption, return_tensors='pt').to('cuda')
                            with torch.inference_mode():
                                result = model(**inputs)
                            detected = processor.post_process_grounded_object_detection(
                                result, inputs.input_ids, threshold=.20, text_threshold=.18,
                                target_sizes=[(image.height,image.width)])[0]
                            labels = detected.get('text_labels', detected.get('labels'))
                            boxes = [dict(label=str(label), score=float(score), box_xyxy=box.tolist())
                                for label,score,box in zip(labels, detected['scores'].cpu(), detected['boxes'].cpu())]
                        else:
                            boxes=[]
                            for role,phrase in roles:
                                task='<CAPTION_TO_PHRASE_GROUNDING>'
                                inputs=processor(text=task+phrase,images=image,return_tensors='pt')
                                with torch.inference_mode():
                                    ids=model.generate(input_ids=inputs['input_ids'].to('cuda'),
                                        pixel_values=inputs['pixel_values'].to(device='cuda',dtype=torch.float16),
                                        max_new_tokens=128,num_beams=1,do_sample=False,use_cache=False)
                                text=processor.batch_decode(ids,skip_special_tokens=False)[0]
                                parsed=processor.post_process_generation(text,task=task,image_size=image.size)[task]
                                boxes.extend(dict(label=str(label),score=None,query_role=role,query=phrase,
                                    box_xyxy=[float(v) for v in box]) for label,box in zip(parsed.get('labels',[]),parsed.get('bboxes',[])))
                        opaque = hashlib.sha256(f'{key}:{row["action_index"]}:{camera}'.encode()).hexdigest()[:16]
                        record = dict(id=opaque, episode_key=key, action_index=row['action_index'], camera=camera,
                            frame_timing='pre_action', task_language=row['task_language'], roles=roles,
                            width=image.width, height=image.height, boxes=boxes,
                            calibrated=False, phase=None)
                        output.write(json.dumps(record, ensure_ascii=False)+'\n')
                        if args.pilot:
                            canvas = image.resize((448,448))
                            draw = ImageDraw.Draw(canvas)
                            for j, box in enumerate(boxes):
                                rect = [2*v for v in box['box_xyxy']]
                                color = ('red','lime','cyan','yellow','magenta')[j%5]
                                draw.rectangle(rect, outline=color, width=2)
                                score_text=f'{box["score"]:.2f}' if box['score'] is not None else 'uncal'
                                draw.text((max(0,rect[0]),max(0,rect[1]-12)), f'{j}:{box["label"]} {score_text}', fill=color)
                            canvas.save(args.output_dir/f'{opaque}.jpg', quality=90)
                        total += 1
                        if total % 20 == 0:
                            output.flush()
                            print(json.dumps(dict(images=total, elapsed_s=round(time.monotonic()-start,1))), flush=True)
                    del images
    summary = dict(model=args.model, revision=getattr(model.config, '_commit_hash', None),
        pilot=args.pilot, backend=args.backend, camera=args.camera,
        negative_controls=args.negative_controls, action_index_filter=args.action_index,
        episodes=len(grouped), images=total, elapsed_s=time.monotonic()-start,
        torch=torch.__version__, device=torch.cuda.get_device_name(),
        purpose='uncalibrated_localization_only_no_accuracy_claim')
    (args.output_dir/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
