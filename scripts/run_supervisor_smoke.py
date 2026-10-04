#!/usr/bin/env python3
"""CPU-only end-to-end smoke run of interruption and replanning."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vla_supervisor.events import RecommendedAction
from vla_supervisor.factory import SupervisorConfig, create_safe_default
from vla_supervisor.monitors import StallConfig
from vla_supervisor.recovery import RecoveryConfig


def state(x): return {"eef_pos":[x,0,0],"joint_pos":[0]*7,"joint_vel":[0]*7}


def main():
    p=argparse.ArgumentParser();p.add_argument("--log",type=Path,required=True);p.add_argument("--summary",type=Path,required=True);a=p.parse_args()
    config=SupervisorConfig(stall=StallConfig(low_progress_run_threshold=2,gripper_reversal_threshold=99),
                            recovery=RecoveryConfig(max_replans=2,cooldown_steps=2))
    runtime=create_safe_default(config,a.log);runtime.reset_episode();runtime.install_chunk([[1,0,0,0,0,0,1]]*8)
    decisions=[]
    for i in range(3):
        action,pre=runtime.next_action(state(0),i);assert action is not None
        decisions.append(runtime.observe_step(intended_action=action,state_before=state(0),state_after=state(0),action_index=i))
    assert decisions[-1].action is RecommendedAction.STOP_AND_REPLAN and not runtime.chunk
    runtime.mark_replan_completed([[.2,0,0,0,0,0,1]]*3)
    # New plan now makes progress and must not immediately re-alarm.
    for i in range(3,6):
        action,_=runtime.next_action(state((i-3)*.01),i);assert action is not None
        decision=runtime.observe_step(intended_action=action,state_before=state((i-3)*.01),
                                      state_after=state((i-2)*.01),action_index=i)
        assert decision.action is RecommendedAction.CONTINUE
    runtime.logger.close()
    result={"status":"PASS","component_status":runtime.component_status,"replans":runtime.recovery.replans,
            "first_alarm_reason":decisions[-1].reason.value,"chunk_interrupted":True,
            "post_replan_steps_completed":3,"utf8_log":str(a.log)}
    a.summary.parent.mkdir(parents=True,exist_ok=True);a.summary.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
