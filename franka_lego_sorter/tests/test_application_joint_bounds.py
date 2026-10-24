import unittest
from types import SimpleNamespace as S
from lego_sorter.application_joint_bounds import validate_application_joint_bounds


def trajectory(q5,q6,velocities=None):
    points=[]
    for i,(a,b)in enumerate(zip(q5,q6)):
        v=[0.,0.]if velocities is None else velocities[i]
        points.append(S(positions=[a,b],velocities=v,accelerations=[0.,0.],time_from_start=S(sec=i,nanosec=0)))
    return S(joint_trajectory=S(joint_names=['fr3_joint5','fr3_joint6'],points=points))


class ApplicationJointBoundsTests(unittest.TestCase):
    def test_open_branch_passes(self):
        self.assertEqual(set(validate_application_joint_bounds(trajectory([0.,.5],[1.,2.]))),{'fr3_joint5','fr3_joint6'})
    def test_flipped_joint5_refused(self):
        with self.assertRaisesRegex(ValueError,'fr3_joint5'):validate_application_joint_bounds(trajectory([0.,2.51],[1.,2.]))
    def test_folded_joint6_refused(self):
        with self.assertRaisesRegex(ValueError,'fr3_joint6'):validate_application_joint_bounds(trajectory([0.,.5],[1.,3.81]))
    def test_between_waypoint_overshoot_refused(self):
        with self.assertRaisesRegex(ValueError,'fr3_joint5'):validate_application_joint_bounds(trajectory([1.4,1.4],[1.,1.],[[2.,0.],[-2.,0.]]))
