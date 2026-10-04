import unittest
from vla_supervisor.embodiment import CartesianDeltaEmbodiment
from vla_supervisor.intervention import InterventionPlanner


class JointLikeTestAdapter:
    adapter_id='test-joint'
    def hold_action(self,last_action): return (9.,8.)
    def retreat_action(self,last_action,magnitude): return (-7.,-6.)


class TestEmbodiment(unittest.TestCase):
    def test_default_layout_preserves_gripper(self):
        a=CartesianDeltaEmbodiment().hold_action([1,2,3,4,5,6,-1])
        self.assertEqual(a,(0.,0.,0.,0.,0.,0.,-1.))

    def test_custom_layout(self):
        e=CartesianDeltaEmbodiment(5,(1,2,3),0,'robot-b')
        self.assertEqual(e.hold_action([.7,1,2,3,4]),(.7,0.,0.,0.,0.))
        r=e.retreat_action([.7,3,0,4,9],.2)
        self.assertAlmostEqual(r[1],-.12);self.assertAlmostEqual(r[3],-.16)

    def test_invalid_layout(self):
        with self.assertRaises(ValueError): CartesianDeltaEmbodiment(2,(0,1,2),None)

    def test_release_can_zero_or_preserve_motion(self):
        e=CartesianDeltaEmbodiment()
        reference=[.2,-.1,.3,.4,.5,.6,1]
        self.assertEqual(e.release_action(reference),(0.,0.,0.,0.,0.,0.,-1.))
        self.assertEqual(
            e.release_action(reference,preserve_motion=True),
            (.2,-.1,.3,.4,.5,.6,-1.),
        )

    def test_planner_uses_adapter(self):
        p=InterventionPlanner(embodiment=JointLikeTestAdapter())
        self.assertEqual(p._hold_action(None),(9.,8.))
        self.assertEqual(p._retreat_action(None),(-7.,-6.))


if __name__=='__main__': unittest.main()
