import unittest
from types import SimpleNamespace as S
from unittest.mock import Mock
import numpy as np
from lego_sorter.object_diagnostic import execute_stage

class ObjectDiagnosticTests(unittest.TestCase):
    def test_unimplemented_contact_or_lift_is_not_implicitly_enabled(self):
        r=S(execute_linear_plan=Mock(),gripper_state=Mock())
        with self.assertRaises(ValueError):execute_stage(r,object(),{'stage':'object_lift_test'})
        r.execute_linear_plan.assert_not_called();r.gripper_state.assert_not_called()
    def test_approach_requires_the_measured_opening(self):
        r=S(execute_linear_plan=Mock(),gripper_state=Mock(return_value={'width_m':.02}))
        with self.assertRaises(RuntimeError):execute_stage(r,object(),{'stage':'object_approach30','gripper_width_m':.03})
        r.execute_linear_plan.assert_not_called()
    def test_no_hand_or_lift_action_is_sent(self):
        r=S(execute_linear_plan=Mock(return_value=np.eye(4)),gripper_state=Mock(return_value={'width_m':.03}))
        out=execute_stage(r,object(),{'stage':'object_approach30','gripper_width_m':.03})
        self.assertFalse(out['gripper_action_sent']);self.assertFalse(out['lift_action_sent']);self.assertFalse(out['production_count_changed'])

    def grasp_fixture(self,d,offset):
        import json
        from pathlib import Path
        goal=np.diag([1.,-1.,-1.,1.]);goal[2,3]=.0465
        p=Path(d)/'target.json';p.write_text(json.dumps({'normal_base':[0,0,1],'visible_upper_p99_offset_base_m':.040,'toe_center_insertion_below_observed_p99_m':.003,'minimum_toe_clearance_above_prior_lower_layer_globalmax_m':.0084}))
        actual=goal.copy();actual[0,3]+=offset
        r=S(execute_linear_plan=Mock(return_value=actual),gripper_state=Mock(return_value={'width_m':.03}))
        m={'stage':'object_grasp_pose','object_target':str(p),'physical_goal':goal.tolist(),'gripper_width_m':.03,'reference_execution_success':False}
        return r,m,goal

    def test_grasp_pose_strict_target_preserves_previous_failure(self):
        import tempfile
        with tempfile.TemporaryDirectory()as d:
            r,m,goal=self.grasp_fixture(d,.0008);out=execute_stage(r,object(),m)
            self.assertFalse(out['reference_execution_success']);self.assertFalse(out['gripper_action_sent'])
            np.testing.assert_allclose(r.execute_linear_plan.call_args.kwargs['verification_target'],goal)

    def test_grasp_pose_does_not_accept_old2mm_tolerance(self):
        import tempfile
        with tempfile.TemporaryDirectory()as d:
            r,m,_=self.grasp_fixture(d,.0015)
            with self.assertRaisesRegex(RuntimeError,'no finger action'):execute_stage(r,object(),m)

    def test_observed_rotation_can_be_partially_compensated_without_raising_command_cap(self):
        from scipy.spatial.transform import Rotation
        from lego_sorter.object_diagnostic import bounded_approach_rotation
        applied,info=bounded_approach_rotation(Rotation.from_rotvec([0,0,.0106]).as_matrix())
        self.assertAlmostEqual(info['observed_approach_rotation_rad'],.0106)
        self.assertAlmostEqual(Rotation.from_matrix(applied).magnitude(),.0095)
        self.assertTrue(info['rotation_feedforward_saturated'])

    def test_large_observed_rotation_is_not_accepted(self):
        from scipy.spatial.transform import Rotation
        from lego_sorter.object_diagnostic import bounded_approach_rotation
        with self.assertRaisesRegex(ValueError,'acquisition'):bounded_approach_rotation(Rotation.from_rotvec([0,0,.021]).as_matrix())

    def test_small_rotation_is_not_amplified(self):
        from scipy.spatial.transform import Rotation
        from lego_sorter.object_diagnostic import bounded_approach_rotation
        applied,_=bounded_approach_rotation(Rotation.from_rotvec([0,.005,0]).as_matrix())
        self.assertAlmostEqual(Rotation.from_matrix(applied).magnitude(),.005)

    def test_grasp_command_sampling_noise_is_bounded(self):
        from scipy.spatial.transform import Rotation
        from lego_sorter.object_diagnostic import sampling_pose_matches
        expected=np.eye(4);sample=expected.copy();sample[0,3]=.00000735;sample[:3,:3]=Rotation.from_rotvec([0,0,.00001483]).as_matrix()
        self.assertTrue(sampling_pose_matches(sample,expected))
        sample[0,3]=.001
        self.assertFalse(sampling_pose_matches(sample,expected))
