import unittest
from vla_supervisor.goal_relations import GoalRelation,SimpleEnglishManipulationGoalAdapter,TaskGoalAdapter


class StructuredAdapter:
    adapter_id='structured-test'
    def parse(self,s): return GoalRelation('insert',s,'socket')


class TestTaskGoalAdapter(unittest.TestCase):
    def test_protocol_accepts_non_language_adapter(self):
        adapter:TaskGoalAdapter=StructuredAdapter()
        self.assertEqual(adapter.parse('peg').relation,'insert')

    def test_demo_parser_marks_unsupported(self):
        r=SimpleEnglishManipulationGoalAdapter().parse('assemble component alpha')
        self.assertEqual(r.relation,'unsupported')

    def test_no_task_id_interface(self):
        r=SimpleEnglishManipulationGoalAdapter().parse('put the cup in the bin')
        self.assertEqual((r.relation,r.manipulated_object,r.target),('place_in','the cup','the bin'))

if __name__=='__main__':unittest.main()
