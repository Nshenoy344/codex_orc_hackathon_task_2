import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import Mock
import numpy as np
from lego_sorter.isolated_lift10 import execute_lift10

class IsolatedLift10Tests(unittest.TestCase):
    def fixture(self,d,up=.010,widths=(.01531,.01531)):
        p=Path(d)/'hold.json';p.write_text(json.dumps({'success':True,'hand_grasp_and_body_width_verified':True,'gripper_after':{'width_m':.01531}}))
        before=np.eye(4);after=before.copy();after[2,3]+=up
        robot=S(gripper_state=Mock(side_effect=[{'width_m':v}for v in widths]),current_pose=Mock(return_value=before),execute_linear_plan=Mock(return_value=after))
        return robot,{'stage':'lift10','hold_result':str(p)}
    def test_one_lift_has_no_fingers_and_no_visual_success_claim(self):
        with tempfile.TemporaryDirectory()as d:
            r,m=self.fixture(d);out=execute_lift10(r,object(),m)
            self.assertTrue(out['success']);self.assertFalse(out['gripper_action_sent']);self.assertFalse(out['brick_lifted_visually_verified']);r.execute_linear_plan.assert_called_once()
    def test_changed_before_hold_prevents_arm_action(self):
        with tempfile.TemporaryDirectory()as d:
            r,m=self.fixture(d,widths=(.014,.014))
            with self.assertRaisesRegex(RuntimeError,'no arm command'):execute_lift10(r,object(),m)
            r.execute_linear_plan.assert_not_called()
    def test_small_actual_rise_does_not_claim_lift(self):
        with tempfile.TemporaryDirectory()as d:
            r,m=self.fixture(d,up=.003)
            with self.assertRaisesRegex(RuntimeError,'pending'):execute_lift10(r,object(),m)
    def test_hold_loss_after_motion_stops(self):
        with tempfile.TemporaryDirectory()as d:
            r,m=self.fixture(d,widths=(.01531,.001))
            with self.assertRaisesRegex(RuntimeError,'pending'):execute_lift10(r,object(),m)
