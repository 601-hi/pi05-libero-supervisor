"""Replay genuine causal inputs with missing vision explicitly represented.

This readiness test cannot evaluate accuracy. It checks that incomplete evidence
never gets silently interpreted as zero error, successful grasp, or failure.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
from vla_supervisor.goal_progress import GoalProgressMonitor, GoalProgressObservation


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    states = Counter()
    actions = Counter()
    episode = None
    monitor = GoalProgressMonitor()
    with args.inputs.open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            if row['episode_key'] != episode:
                monitor.reset()
                episode = row['episode_key']
            if not row['visual_query_due']:
                continue
            if row['target_measurement'] is not None:
                raise ValueError('This readiness replay requires an explicit measured-vision adapter before using non-null measurements')
            result = monitor.observe(GoalProgressObservation(
                action_index=row['action_index'], phase=row['phase'] or 'unknown', target_id='',
                measurement_domain='unverified', metric_kind='image_distance',
                goal_error=None, error_bound=None, eef_position=tuple(row['eef_position']),
                observable=False, identity_reliable=False, phase_reliable=False, domain_verified=False))
            states[result.state] += 1
            actions[result.shadow_event().recommended_action.value] += 1
    result = dict(status='readiness_only_no_accuracy_claim', observations=sum(states.values()),
        states=dict(states), recommended_actions=dict(actions),
        calibration='absent', vision_measurements='absent')
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
