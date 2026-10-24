"""Regression checks for the narrowly bounded historical joint4 start exception."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

TOOLS=Path(__file__).resolve().parents[1]/'tools'
spec=importlib.util.spec_from_file_location('validate_lift_plan',TOOLS/'validate_lift_plan.py')
validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)


def review(qs):
    def pose(q):
        t=np.eye(4);t[:3,:3]=Rotation.from_rotvec([0,0,q]).as_matrix();t[2,3]=.35;return t.tolist()
    data=dict(base_frame='base',tcp_frame='tcp',fraction=1.,joint_names=['fr3_joint4'],
              T_base_tcp_start=pose(qs[0]),T_base_tcp_target=pose(qs[-1]),
              points=[dict(positions=[q],velocities=[0.],accelerations=[0.],time_sec=float(i*4))
                      for i,q in enumerate(qs)])
    urdf='''<robot name="unit"><link name="base"/><link name="tcp"/>
    <joint name="fr3_joint4" type="revolute"><parent link="base"/><child link="tcp"/>
    <origin xyz="0 0 .35"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="-.1" velocity="2" effort="10"/></joint></robot>'''
    with tempfile.TemporaryDirectory()as d:
        p=Path(d);(p/'plan.json').write_text(json.dumps(data));(p/'robot.urdf').write_text(urdf)
        return validator.validate(p/'plan.json',p/'robot.urdf')


class LiftValidationTests(unittest.TestCase):
    def test_valid_interior_joint4_may_change_direction(self):
        result=review([-2.,-2.005,-1.998])
        joint=result['joint_limits']['fr3_joint4']
        self.assertEqual(joint['initial_below_lower_rad'],0.)
        self.assertFalse(joint['initial_recovering_exception'])
        self.assertTrue(joint['passed'])

    def test_actual_small_initial_excursion_requires_monotonic_recovery(self):
        joint=review([-3.0001,-3.,-2.9999])['joint_limits']['fr3_joint4']
        self.assertTrue(joint['initial_recovering_exception']);self.assertTrue(joint['passed'])
        bad=review([-3.0001,-3.00012,-2.9999])['joint_limits']['fr3_joint4']
        self.assertTrue(bad['initial_recovering_exception']);self.assertFalse(bad['passed'])

    def test_large_initial_excursion_never_gets_exception(self):
        joint=review([-3.0002,-3.,-2.9999])['joint_limits']['fr3_joint4']
        self.assertFalse(joint['initial_recovering_exception']);self.assertFalse(joint['passed'])


if __name__=='__main__':unittest.main()
