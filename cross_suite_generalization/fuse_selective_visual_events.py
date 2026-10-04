"""Retrospective same-wave fusion of four-state execution and visual events."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from reliability_sequence_supervisor import ReliabilityCalibration


def main():
    p=argparse.ArgumentParser();p.add_argument('--scores',type=Path,required=True);p.add_argument('--visual',type=Path,required=True);p.add_argument('--private-map',type=Path,required=True);p.add_argument('--trace-map',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    score=np.load(a.scores,allow_pickle=False);cal=ReliabilityCalibration.from_json(a.calibration);private={r['anonymous_id']:r for r in json.loads(a.private_map.read_text(encoding='utf-8'))['records']};files={(r['suite'],int(r['task_id'])):int(r['file_index']) for r in json.loads(a.trace_map.read_text(encoding='utf-8'))['files']};visual={r['anonymous_id']:r for r in json.loads(a.visual.read_text(encoding='utf-8'))['records']};rows=[]
    for identifier,meta in private.items():
        fi=files[(meta['suite'],int(meta['task_id']))];mask=(score['file_index']==fi)&(score['episode_idx']==int(meta['episode_idx']));indices=np.flatnonzero(mask);states={}
        for i in indices:
            states[int(score['action_index'][i])]=cal.score(float(score['normal_logp'][i]),float(score['abnormal_logp'][i]),float(score['ensemble_std'][i]),1.0)['decision']
        action_alarm=any(x=='known_abnormal' for x in states.values());events=[]
        recoveries=[e for e in visual[identifier]['events'] if e['type']=='attachment_recovered']
        recovery_latency=None if not recoveries else int(recoveries[0]['latency_from_instability'])
        for event in visual[identifier]['events']:
            if event['type']!='unexpected_attachment_loss':continue
            ai=None if event.get('command') is None else int(event['command']['action_index']);state=states.get(ai,'unscored');routed=state in {'ambiguous_overlap','unknown'};events.append({**event,'execution_state':state,'routed_visual_alarm':routed})
        visual_alarm=any(e['routed_visual_alarm'] for e in events)
        sustained_visual_alarm=visual_alarm and (recovery_latency is None or recovery_latency>3)
        rows.append({'anonymous_id':identifier,'outcome_posthoc_only':meta['outcome_for_posthoc_audit_only'],'action_only_alarm':action_alarm,'routed_visual_alarm':visual_alarm,'recovery_latency':recovery_latency,'sustained_visual_alarm_diagnostic':sustained_visual_alarm,'combined_alarm':action_alarm or visual_alarm,'combined_sustained_alarm_diagnostic':action_alarm or sustained_visual_alarm,'visual_events':events})
    def summary(field):
        return {'failure_detected':sum(r['outcome_posthoc_only']=='failure' and r[field] for r in rows),'failure_total':sum(r['outcome_posthoc_only']=='failure' for r in rows),'success_alarm':sum(r['outcome_posthoc_only']=='success' and r[field] for r in rows),'success_total':sum(r['outcome_posthoc_only']=='success' for r in rows),'episode_accuracy':sum((r['outcome_posthoc_only']=='failure')==r[field] for r in rows)/len(rows)}
    result={'schema_version':1,'evaluation_role':'retrospective diagnostic; Wave3 was previously opened','fusion_rule':'known abnormal OR (unexpected attachment loss AND execution ambiguous/unknown at same action)','wave4_candidate_rule':'suppress attachment instability that recovers at the minimum three-frame confirmation latency; not an independent Wave3 claim','summary':{'action_only':summary('action_only_alarm'),'selective_combined':summary('combined_alarm'),'selective_combined_sustained_diagnostic':summary('combined_sustained_alarm_diagnostic'),'routed_visual_episode_alarms':sum(r['routed_visual_alarm'] for r in rows)},'records':rows}
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result['summary'],indent=2))
if __name__=='__main__':main()
