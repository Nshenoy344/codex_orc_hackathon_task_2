import types
import unittest
from unittest.mock import Mock
import numpy as np
from lego_sorter.endpoint_servo import (plan_endpoint_correction, execute_endpoint_correction,
                                        EndpointServoError)


def rz(a):
    c,s=np.cos(a),np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1.]])


class FakeRobot:
    def __init__(self, bias=(.0007,.0004,.002), angle=.0048, converges=True):
        self.cfg=dict(position_tolerance_m=.002,rotation_tolerance_rad=.025)
        self.goal=np.eye(4);self.goal[:3,3]=[.24,-.01,.29]
        self.bias=np.array(bias);self.rotation_bias=rz(angle)
        self.actual=self.goal.copy();self.actual[:3,3]-=self.bias
        self.actual[:3,:3]=self.rotation_bias.T@self.goal[:3,:3]
        self.plans=[];self.executions=[];self.converges=converges
    def current_pose(self):return self.actual.copy()
    def plan_linear(self,target):
        plan=types.SimpleNamespace(target_pose=target.copy(),noop=False,
                                  to_dict=lambda:dict(T_base_tcp_target=target.tolist()))
        self.plans.append(plan);return plan
    def execute_linear_plan(self,plan,verification_target=None):
        self.executions.append((plan,verification_target.copy()))
        if self.converges:
            self.actual=plan.target_pose.copy();self.actual[:3,3]-=self.bias
            self.actual[:3,:3]=self.rotation_bias.T@self.actual[:3,:3]
        return self.actual.copy()


class EndpointServoTest(unittest.TestCase):
    def test_one_corrected_command_verifies_original_physical_goal(self):
        robot=FakeRobot()
        correction=plan_endpoint_correction(robot,robot.goal)
        self.assertEqual(len(robot.plans),1);self.assertEqual(len(robot.executions),0)
        np.testing.assert_allclose(correction.command_target[:3,3],robot.goal[:3,3]+robot.bias)
        report=execute_endpoint_correction(robot,correction)
        self.assertTrue(report['success']);self.assertEqual(report['attempts'],1)
        self.assertLess(report['residual_position_error_m'],1e-12)
        np.testing.assert_allclose(robot.executions[0][1],robot.goal)
        self.assertFalse(np.allclose(correction.command_target,robot.goal))

    def test_already_converged_does_not_plan_or_execute(self):
        robot=FakeRobot(bias=(.0001,0,.0001),angle=0)
        self.assertIsNone(plan_endpoint_correction(robot,robot.goal))
        self.assertEqual(robot.plans,[]);self.assertEqual(robot.executions,[])

    def test_excessive_correction_refused_without_plan(self):
        robot=FakeRobot(bias=(0,0,.004))
        with self.assertRaisesRegex(EndpointServoError,'outside correction bound'):
            plan_endpoint_correction(robot,robot.goal)
        self.assertEqual(robot.plans,[])

    def test_downward_offset_refused(self):
        robot=FakeRobot(bias=(0,0,-.002))
        with self.assertRaisesRegex(EndpointServoError,'downward'):
            plan_endpoint_correction(robot,robot.goal)
        self.assertEqual(robot.plans,[])

    def test_failed_single_correction_never_auto_retries(self):
        robot=FakeRobot(converges=False)
        correction=plan_endpoint_correction(robot,robot.goal)
        with self.assertRaisesRegex(EndpointServoError,'did not converge'):
            execute_endpoint_correction(robot,correction)
        self.assertEqual(len(robot.plans),1);self.assertEqual(len(robot.executions),1)

    def test_tolerance_cannot_be_loosened(self):
        robot=FakeRobot()
        with self.assertRaisesRegex(ValueError,'no looser'):
            plan_endpoint_correction(robot,robot.goal,dict(position_tolerance_m=.003))
        self.assertEqual(robot.plans,[])

    def test_rotation_ceiling_checked_before_planning(self):
        robot=FakeRobot(bias=(0,0,.002),angle=.012)
        with self.assertRaisesRegex(EndpointServoError,'outside correction bound'):
            plan_endpoint_correction(robot,robot.goal)
        self.assertEqual(robot.plans,[])


if __name__=='__main__':unittest.main()
