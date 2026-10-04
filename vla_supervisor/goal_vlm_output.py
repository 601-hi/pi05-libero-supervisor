"""Strict parsing of untrusted VLM measurements; never calibrated evidence."""
import json
import math


def parse_goal_vlm(text):
    cleaned=text.strip()
    if cleaned.startswith('```') and cleaned.endswith('```'):
        cleaned='\n'.join(cleaned.splitlines()[1:-1])
    try:
        value=json.loads(cleaned)
    except (ValueError,TypeError):
        return {'valid':False,'reason':'invalid_json','calibrated':False}
    if not isinstance(value,dict):
        return {'valid':False,'reason':'not_object','calibrated':False}
    required=('gripper_center','object_box','destination_box','phase','goal_state','evidence')
    if any(k not in value for k in required):
        return {'valid':False,'reason':'missing_fields','calibrated':False}
    for name,size in [('gripper_center',2),('object_box',4),('destination_box',4)]:
        coords=value[name]
        if coords is None:
            continue
        if (not isinstance(coords,list) or len(coords)!=size or
            any(type(x) not in (int,float) or not math.isfinite(x) or not 0<=x<=1000 for x in coords)):
            return {'valid':False,'reason':'invalid_coordinates_'+name,'calibrated':False}
        if size==4 and not (coords[0]<coords[2] and coords[1]<coords[3]):
            return {'valid':False,'reason':'inverted_box_'+name,'calibrated':False}
    if value['phase'] not in {'approach','grasp','transport','place','manipulate','retract','unknown'}:
        return {'valid':False,'reason':'invalid_phase','calibrated':False}
    if value['goal_state'] not in {'achieved','not_achieved','unknown'} or not isinstance(value['evidence'],str):
        return {'valid':False,'reason':'invalid_state_or_evidence','calibrated':False}
    return {'valid':True,'calibrated':False,'coordinates':'fixed_image_normalized_0_1000',
            **{k:value[k] for k in required}}
