import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import Mock
import numpy as np
from lego_sorter.isolated_grasp_pose import execute_grasp_pose

class IsolatedGraspPoseTests(unittest.TestCase):
    def fixture(self,d,offset):
        goal=np.diag([1.,-1.,-1.,1.]);goal[2,3]=-.003
        path=Path(d)/'targets.json';path.write_text(json.dumps({'planes':[{'plane':'left','measured_table_plane':{'normal_base':[0,0,1]},'floor_ray_point_base_m':[0,0,-.020]}]}))
        meta={'stage':'grasp_pose','physical_goal':goal.tolist(),'targets':str(path),'plane':'left'}
        actual=goal.copy();actual[0,3]+=offset
        robot=S(execute_linear_plan=Mock(return_value=actual));return robot,meta,goal
    def test_verifies_original_physical_goal_and_has_no_finger_action(self):
        with tempfile.TemporaryDirectory()as d:
            robot,meta,goal=self.fixture(d,.0008);plan=object();result=execute_grasp_pose(robot,plan,meta)
            self.assertTrue(result['success']);self.assertFalse(result['gripper_action_sent']);self.assertFalse(result['lift_action_sent'])
            args=robot.execute_linear_plan.call_args
            self.assertIs(args.args[0],plan);np.testing.assert_allclose(args.kwargs['verification_target'],goal)
    def test_old2mm_tolerance_is_not_accepted_before_closing(self):
        with tempfile.TemporaryDirectory()as d:
            robot,meta,_=self.fixture(d,.0015)
            with self.assertRaisesRegex(RuntimeError,'no finger action'):execute_grasp_pose(robot,object(),meta)
    def test_actual_tip_floor_clearance_is_checked(self):
        with tempfile.TemporaryDirectory()as d:
            robot,meta,goal=self.fixture(d,0.);goal[2,3]=-.0085;meta['physical_goal']=goal.tolist();robot.execute_linear_plan.return_value=goal
            with self.assertRaisesRegex(RuntimeError,'no finger action'):execute_grasp_pose(robot,object(),meta)
    def test_unrelated_stage_refused_without_motion(self):
        robot=S(execute_linear_plan=Mock())
        with self.assertRaises(ValueError):execute_grasp_pose(robot,object(),{'stage':'grasp'})
        robot.execute_linear_plan.assert_not_called()
