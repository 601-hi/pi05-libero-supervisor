"""CPU-only descriptive audit. Motion efficiency is NOT goal progress."""
import argparse
import collections
import json
from pathlib import Path
import numpy as np


def window_metrics(rows):
    before = np.asarray([r['eef_pos_before'] for r in rows], float)
    after = np.asarray([r['eef_pos_after'] for r in rows], float)
    delta = after - before
    lengths = np.linalg.norm(delta, axis=1)
    path = float(lengths.sum())
    net = float(np.linalg.norm(after[-1] - before[0]))
    valid_pairs = (lengths[:-1] > 1e-5) & (lengths[1:] > 1e-5)
    reversals = (np.sum(delta[:-1] * delta[1:], axis=1) < 0) & valid_pairs
    return dict(path_m=path, net_m=net,
                net_over_path=net/path if path > 1e-9 else None,
                translation_pause_fraction=float(np.mean(lengths < 0.0005)),
                direction_reversal_count=int(reversals.sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.manifest.read_text(encoding='utf-8'))
    cache = {}
    episodes = []
    for episode in source['episodes']:
        path = episode['trace_path']
        if path not in cache:
            grouped = collections.defaultdict(list)
            with open(path, encoding='utf-8') as handle:
                for line in handle:
                    row = json.loads(line)
                    if row.get('event') == 'step':
                        grouped[(row['task_id'], row['episode_idx'])].append(row)
            cache[path] = grouped
        rows = cache[path][(episode['task_id'], episode['episode_idx'])]
        indices = [r['action_index'] for r in rows]
        with np.load(episode['sidecar_path'], allow_pickle=False) as sidecar:
            # Headers only: do not decompress image arrays on a small CPU instance.
            import zipfile
            headers = {}
            with zipfile.ZipFile(episode['sidecar_path']) as archive:
                for name in archive.namelist():
                    with archive.open(name) as member:
                        version = np.lib.format.read_magic(member)
                        shape, _, dtype = np.lib.format._read_array_header(member, version)
                        headers[name.removesuffix('.npy')] = dict(shape=list(shape), dtype=str(dtype))
            sidecar_indices = sidecar['action_indices'].tolist()
        frame_indices = [r.get('visual_frame_index') for r in rows]
        aligned = all(isinstance(f, int) and 0 <= f < len(sidecar_indices)
                      and sidecar_indices[f] == r['action_index']
                      for f, r in zip(frame_indices, rows))
        windows = []
        for width in (10, 20, 40):
            for end in range(width - 1, len(rows)):
                block = rows[end-width+1:end+1]
                windows.append(dict(width=width, end_action_index=indices[end], **window_metrics(block)))
        episodes.append(dict(
            episode_id=episode['episode_id'], suite=episode['suite'], task_id=episode['task_id'],
            success=episode['offline_evaluation_only']['episode_success'],
            steps=len(rows), goal_family=episode['runtime_inputs']['goal_family_from_language'],
            task_language=episode['runtime_inputs']['task_language'],
            contiguous_actions=indices == list(range(indices[0], indices[0]+len(indices))),
            frame_indices_complete=all(isinstance(i, int) for i in frame_indices),
            exact_frame_action_alignment=aligned,
            both_camera_lengths_match=all(headers[k]['shape'][0] == len(sidecar_indices)
                                          for k in ('agent_images', 'wrist_images')),
            step_fields=sorted(set.intersection(*(set(r) for r in rows))),
            sidecar_headers=headers, **window_metrics(rows), windows=windows))
    summaries = {}
    for success in (True, False):
        selected = [e for e in episodes if e['success'] == success]
        summaries[str(success)] = dict(episodes=len(selected), steps=sum(e['steps'] for e in selected))
        for width in (10, 20, 40):
            # Descriptive episodes, not independent-window confidence estimates.
            rates = []
            for e in selected:
                windows = [w for w in e['windows'] if w['width'] == width]
                if windows:
                    rates.append(sum(w['path_m'] > .01 and w['net_over_path'] < .3 for w in windows)/len(windows))
            summaries[str(success)][str(width)] = dict(
                episodes_with_looplike_window=sum(r > 0 for r in rates),
                episode_fraction_quantiles=np.quantile(rates, [0,.5,.9,1]).tolist() if rates else [])
    result = dict(status='descriptive_development_audit_not_detector',
                  warning='No measured target positions or stage labels; net/path is not target progress. Thresholds are illustrative, not calibrated alarms. Success does not label every step normal.',
                  summaries=summaries, episodes=episodes)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
