"""Export causal pre-action kinematics and anonymous visual lookup for seed36.

The path lookup is I/O metadata, never a vision-model input: original filenames
can contain outcome labels. Consumers pass pixels plus runtime fields only.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from vla_supervisor.goal_relations import parse_goal_relation
from vla_supervisor.goal_progress import GoalProgressQuerySchedule
from cross_suite_generalization.build_semantic_consequence_manifest import runtime_family


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    source = json.loads(args.manifest.read_text(encoding='utf-8'))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache, lookup, coverage = {}, {}, []
    counts = dict(episodes=0, steps=0, periodic_queries=0, unsupported_goal_episodes=0,
                  missing_target_measurements=0, pose_chain_mismatches=0,
                  corrected_goal_family_episodes=0)
    with (args.output_dir/'causal_inputs.jsonl').open('w', encoding='utf-8') as out:
        for ep in source['episodes']:
            path = ep['trace_path']
            if path not in cache:
                with open(path, encoding='utf-8') as handle:
                    cache[path] = [json.loads(line) for line in handle if line.strip()]
            rows = [r for r in cache[path] if r.get('event') == 'step'
                    and r['task_id'] == ep['task_id'] and r['episode_idx'] == ep['episode_idx']]
            identity = hashlib.sha256(('seed36:'+ep['episode_id']).encode()).hexdigest()[:16]
            lookup[identity] = {'sidecar_path': ep['sidecar_path'], 'trace_path': path,
                                'episode_id': ep['episode_id']}
            relation = parse_goal_relation(ep['runtime_inputs']['task_language'])
            corrected_family = runtime_family(ep['runtime_inputs']['task_language'])
            counts['corrected_goal_family_episodes'] += corrected_family != ep['runtime_inputs']['goal_family_from_language']
            ep['runtime_inputs']['goal_family_from_language'] = corrected_family
            scheduler = GoalProgressQuerySchedule()
            counts['episodes'] += 1
            counts['unsupported_goal_episodes'] += relation.relation == 'unsupported'
            coverage.append(dict(episode_key=identity, goal=asdict(relation),
                language=ep['runtime_inputs']['task_language']))
            for i, row in enumerate(rows):
                due = scheduler.due(row['action_index'])
                counts['steps'] += 1
                counts['periodic_queries'] += due
                counts['missing_target_measurements'] += 1
                if i and not np.allclose(rows[i-1]['eef_pos_after'], row['eef_pos_before'], rtol=0, atol=1e-10):
                    counts['pose_chain_mismatches'] += 1
                runtime = dict(episode_key=identity, action_index=row['action_index'],
                    image_frame_index=row['visual_frame_index'], frame_timing='pre_action',
                    eef_position=row['eef_pos_before'], eef_quaternion=row['eef_quat_before'],
                    joint_position=row['joint_pos_before'], joint_velocity=row['joint_vel_before'],
                    gripper_position=row['gripper_qpos_before'],
                    task_language=ep['runtime_inputs']['task_language'], goal=asdict(relation),
                    visual_query_due=due, target_measurement=None, phase=None)
                out.write(json.dumps(runtime, ensure_ascii=False)+'\n')
    for name, payload in [('io_lookup_not_model_input.json', lookup),
                          ('language_coverage.json', coverage), ('summary.json', counts)]:
        (args.output_dir/name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    source['language_parser_revision'] = 'destination_relation_v2_20260920'
    (args.output_dir/'semantic_manifest_language_corrected_v2.json').write_text(
        json.dumps(source, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(counts), flush=True)


if __name__ == '__main__':
    main()
