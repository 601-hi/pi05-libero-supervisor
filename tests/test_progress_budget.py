import math
import unittest
from dataclasses import replace
from vla_supervisor.progress_budget import ProgressEnvelope,ProgressWindow,assess_progress_budget


class TestProgressBudget(unittest.TestCase):
    def setUp(self):
        self.e=ProgressEnvelope('success-cal-v1','approach_relation',.01,.2,.05,40,.01)
        self.w=ProgressWindow('approach_relation',.5,.3,.01,.01,.3,10,20,40,True)

    def test_on_track(self): self.assertEqual(assess_progress_budget(self.w,self.e).state,'on_track')
    def test_short_wander_is_not_stage_failure(self):
        r=assess_progress_budget(replace(self.w,end_error=.51,stage_elapsed_actions=10),self.e)
        self.assertEqual(r.state,'budget_infeasible')
    def test_stage_stall(self):
        self.assertEqual(assess_progress_budget(replace(self.w,stage_elapsed_actions=41),self.e).state,'stage_stalled')
    def test_positive_but_too_slow(self):
        r=assess_progress_budget(replace(self.w,end_error=.45,remaining_action_budget=100),self.e)
        self.assertEqual(r.state,'slow_or_inefficient')
    def test_optimistic_timeout(self):
        r=assess_progress_budget(replace(self.w,end_error=.45,remaining_action_budget=5),self.e)
        self.assertEqual(r.state,'budget_infeasible');self.assertGreater(r.optimistic_actions_to_goal,5)
    def test_uncertainty_can_prevent_wandering_claim(self):
        r=assess_progress_budget(replace(self.w,end_error=.51,start_error_bound=.1,end_error_bound=.1,
            remaining_action_budget=100),self.e)
        self.assertNotEqual(r.state,'wandering_or_away')
    def test_untrusted_abstains(self):
        self.assertEqual(assess_progress_budget(replace(self.w,measurement_trusted=False),self.e).state,'unknown')
    def test_relation_mismatch(self):
        self.assertEqual(assess_progress_budget(replace(self.w,relation_type='transport'),self.e).state,'unknown')
    def test_no_calibration(self): self.assertEqual(assess_progress_budget(self.w,None).state,'unknown')
    def test_inside_band_not_success(self):
        r=assess_progress_budget(replace(self.w,end_error=.02),self.e)
        self.assertEqual(r.state,'within_relation_band');self.assertIn('not necessarily task success',r.reason)
    def test_bad_progress_is_shadow_only(self):
        r=assess_progress_budget(replace(self.w,end_error=.45,remaining_action_budget=5),self.e)
        event=r.to_shadow_event(action_index=20)
        self.assertEqual(event.event_type.value,'relation_progress_inadequate')
        self.assertEqual(event.recommended_action.value,'request_more_evidence')
        self.assertTrue(event.evidence['shadow_only'])
    def test_on_track_does_not_intervene(self):
        event=assess_progress_budget(self.w,self.e).to_shadow_event(action_index=20)
        self.assertEqual(event.event_type.value,'normal')
        self.assertEqual(event.recommended_action.value,'continue')

if __name__=='__main__':unittest.main()
