"""Development-only, outcome-blind window selection and dual-camera review pack.

Manifests containing outcomes/paths stay in the offline audit. Anonymous review
cards contain task language and images only; no success labels or model scores.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
from pathlib import Path
import textwrap
import numpy as np
from PIL import Image, ImageDraw


def select_windows(episode, width=20):
    windows = [w for w in episode['windows'] if w['width'] == width]
    eligible = [w for w in windows if w['path_m'] > .01 and w['net_over_path'] is not None]
    selected = []
    if eligible:
        selected.append(('low_efficiency', min(eligible, key=lambda w: (w['net_over_path'], w['end_action_index']))))
    # A pre-fixed time sample, with no outcome or score-dependent matching.
    if windows:
        selected.append(('fixed_early_reference', windows[min(20, len(windows)-1)]))
    return [dict(reason=reason, start_action_index=w['end_action_index']-width+1, **w)
            for reason, w in selected]


def summarize(episodes):
    result = {}
    for success in (True, False):
        selected = [e for e in episodes if e['success'] is success]
        rates = []
        for e in selected:
            windows = [w for w in e['windows'] if w['width'] == 20 and w['end_action_index'] < 60]
            if len(windows) != 41:
                continue  # Fixed exposure, report exclusions.
            rates.append(sum(w['path_m'] > .01 and w['net_over_path'] < .3 for w in windows)/41)
        result[str(success)] = dict(episodes=len(selected), eligible_prefix60=len(rates),
            any_looplike=sum(r > 0 for r in rates),
            fraction_quantiles=np.quantile(rates, [0,.5,.9,1]).tolist() if rates else [])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding='utf-8'))
    source = json.loads(args.manifest.read_text(encoding='utf-8'))
    lookup = {e['episode_id']: e for e in source['episodes']}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    audit_map = []
    for number, episode in enumerate(audit['episodes']):
        source_episode = lookup[episode['episode_id']]
        windows = select_windows(episode)
        with np.load(source_episode['sidecar_path'], allow_pickle=False) as archive:
            action_indices = archive['action_indices']
            indices = {int(a): i for i, a in enumerate(action_indices)}
            selected_frames = [np.linspace(w['start_action_index'], w['end_action_index'], 5, dtype=int).tolist()
                               for w in windows]
            # Decompress one camera at a time, keeping only selected frames.
            views = {}
            for camera in ('agent_images', 'wrist_images'):
                images = archive[camera]
                views[camera] = [images[[indices[a] for a in frames]].copy() for frames in selected_frames]
                del images
        for slot, (window, frames) in enumerate(zip(windows, selected_frames)):
            card_id = hashlib.sha256(f"seed36:{episode['episode_id']}:{slot}".encode()).hexdigest()[:12]
            canvas = Image.new('RGB', (1120, 528), 'white')
            draw = ImageDraw.Draw(canvas)
            draw.text((6, 4), card_id + ' | top=fixed, bottom=wrist | sparse frames, no outcome label', fill='black')
            draw.text((6, 20), '\n'.join(textwrap.wrap(episode['task_language'], 140)), fill='black')
            for column, action in enumerate(frames):
                draw.text((column*224+4, 58), f'action {action} (PRE action)', fill='black')
                for view, camera in enumerate(('agent_images', 'wrist_images')):
                    canvas.paste(Image.fromarray(views[camera][slot][column]), (column*224, 80+view*224))
            canvas.save(args.output_dir / f'{card_id}.jpg', quality=88)
            cards.append(dict(card_id=card_id, image=f'{card_id}.jpg', task_language=episode['task_language'],
                action_indices=frames, annotation=dict(phase=None, target_visible=None,
                gripper_visible=None, observed_progress=None, ambiguity_reason=None,
                annotation_status='unreviewed')))
            audit_map.append(dict(card_id=card_id, episode_id=episode['episode_id'],
                success=episode['success'], selection=window))
        if (number+1) % 10 == 0:
            print(f'episodes_rendered={number+1}', flush=True)
    groups = collections.defaultdict(list)
    for e in audit['episodes']:
        groups[e['suite']].append(e)
    payload = dict(stage='development_only', window_selection='outcome_blind_retrospective_review_not_online_rule',
        warning='Whole-episode minimum uses future frames. Never use selection time as online detection latency.',
        equal_exposure_prefix60=summarize(audit['episodes']),
        by_suite_prefix60={k:summarize(v) for k,v in groups.items()}, offline_card_map=audit_map)
    (args.output_dir/'offline_audit.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output_dir/'review_cards.json').write_text(json.dumps(cards, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(cards=len(cards), equal_exposure=payload['equal_exposure_prefix60'])), flush=True)


if __name__ == '__main__':
    main()
