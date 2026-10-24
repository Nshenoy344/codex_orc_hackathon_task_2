"""ROS message contract tests; no nodes, DDS sockets, or robot commands."""
import threading
import time
import types
import unittest
from unittest.mock import Mock, patch
import numpy as np

from lego_sorter.motion import RosRobot, LinearPlan, MotionError, ROSResponseTimeout, _matrix, _quaternion, _from_tf

try:
    from geometry_msgs.msg import TransformStamped
    from sensor_msgs.msg import JointState
    from moveit_msgs.action import MoveGroup, ExecuteTrajectory
    from moveit_msgs.msg import MoveItErrorCodes
    from moveit_msgs.srv import GetCartesianPath
    from trajectory_msgs.msg import JointTrajectoryPoint
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False


def adapter():
    robot = RosRobot.__new__(RosRobot)
    robot.cfg = dict(RosRobot.DEFAULTS)
    robot.cfg['state_timeout_sec'] = .08
    robot.cfg['planning_service_discovery_settle_sec'] = 0.
    robot._state_lock = threading.Lock()
    robot._closed = False
    robot._state = None
    robot._robot_state = None
    return robot


class TransformTests(unittest.TestCase):
    def test_reject_nonrigid_pose(self):
        t = np.eye(4)
        t[0, 0] = 2
        with self.assertRaises(ValueError):
            _matrix(t)

    def test_quaternion_at_pi(self):
        q = _quaternion(np.diag([1., -1., -1.]))
        self.assertAlmostEqual(abs(q[0]), 1.)
        self.assertAlmostEqual(np.linalg.norm(q), 1.)


@unittest.skipUnless(ROS_AVAILABLE, 'Source ROS 2 Humble for ROS message contract checks')
class MotionContractTests(unittest.TestCase):
    def test_pose_goal_and_measured_verification(self):
        robot = adapter()
        robot._MoveGroup, robot._move = MoveGroup, None
        target = np.eye(4)
        target[:3, 3] = [.4, 0, .2]
        robot.current_pose = Mock(return_value=target)
        robot._action = Mock(return_value=types.SimpleNamespace(error_code=MoveItErrorCodes(val=1)))
        np.testing.assert_allclose(robot.move_pose(target), target)
        goal = robot._action.call_args.args[1]
        self.assertEqual(goal.request.group_name, 'fr3_arm')
        self.assertFalse(goal.planning_options.plan_only)
        pc = goal.request.goal_constraints[0].position_constraints[0]
        self.assertEqual(pc.link_name, 'fr3_hand_tcp')
        self.assertAlmostEqual(pc.constraint_region.primitive_poses[0].position.x, .4)

    def test_legacy_pose_goal_has_application_wrist_path_bounds(self):
        robot = adapter()
        robot._MoveGroup, robot._move = MoveGroup, None
        target = np.eye(4)
        robot.current_pose = Mock(return_value=target)
        robot._action = Mock(return_value=types.SimpleNamespace(error_code=MoveItErrorCodes(val=1)))
        robot.move_pose(target)
        goal = robot._action.call_args.args[1]
        limits = {c.joint_name: (c.position-c.tolerance_below,
                                c.position+c.tolerance_above)
                  for c in goal.request.path_constraints.joint_constraints}
        self.assertEqual(limits, {'fr3_joint5': (-1.5, 1.5), 'fr3_joint6': (0., 3.)})
        self.assertFalse(goal.planning_options.replan)

    def test_pose_execution_error_rejected(self):
        robot = adapter()
        robot._MoveGroup, robot._move = MoveGroup, None
        robot._action = Mock(return_value=types.SimpleNamespace(error_code=MoveItErrorCodes(val=-1)))
        robot.current_pose = Mock()
        with self.assertRaisesRegex(MotionError, 'MoveIt error'):
            robot.move_pose(np.eye(4))
        robot.current_pose.assert_not_called()

    def test_partial_cartesian_path_is_never_executed(self):
        robot = adapter()
        robot._GetCartesianPath = GetCartesianPath
        robot._ExecuteTrajectory = ExecuteTrajectory
        robot.current_pose = Mock(return_value=np.eye(4))
        joints = JointState()
        joints.name = [f'fr3_joint{i}' for i in range(1, 8)]
        joints.position = [0.] * 7
        robot._robot_state = joints, time.monotonic()
        robot.gripper_state = Mock(return_value={'width_m': .020, 'joints': {'fr3_finger_joint1': .010, 'fr3_finger_joint2': .010}})
        robot._cartesian = Mock()
        robot._cartesian.wait_for_service.return_value = True
        robot._wait = Mock(return_value=types.SimpleNamespace(
            error_code=MoveItErrorCodes(val=1), fraction=.8))
        robot._action = Mock()
        with self.assertRaisesRegex(MotionError, 'Incomplete Cartesian path'):
            target = np.eye(4)
            target[2, 3] = .01
            robot.move_linear(target)
        robot._action.assert_not_called()
        request = robot._cartesian.call_async.call_args.args[0]
        self.assertTrue(request.avoid_collisions)
        self.assertEqual(request.start_state.joint_state.name[:7], joints.name)
        self.assertEqual(request.start_state.joint_state.name[7:], ['fr3_finger_joint1', 'fr3_finger_joint2'])
        self.assertEqual(list(request.start_state.joint_state.position[7:]), [.010, .010])
        self.assertGreater(request.jump_threshold, 0)
        self.assertEqual(robot._cartesian.call_async.call_count,1)

    def test_stale_arm_state_rejected_before_planning(self):
        robot = adapter()
        robot.current_pose = Mock(return_value=np.eye(4))
        robot._cartesian = Mock()
        with self.assertRaisesRegex(MotionError, 'fresh robot joint state'):
            target = np.eye(4)
            target[2, 3] = .01
            robot.move_linear(target)
        robot._cartesian.call_async.assert_not_called()

    def test_initial_arm_discovery_waits_for_fresh_message(self):
        robot = adapter()
        message = JointState()
        message.name = ['fr3_joint1']
        message.position = [0.]
        timer = threading.Timer(.02, lambda: robot._on_robot_state(message))
        timer.start()
        try:
            snapshot = robot._fresh_arm_state()
            self.assertIs(snapshot[0], message)
            self.assertLess(time.monotonic()-snapshot[1], .08)
        finally:
            timer.join()

    def _planned_fixture(self):
        robot = adapter()
        robot._GetCartesianPath, robot._ExecuteTrajectory = GetCartesianPath, ExecuteTrajectory
        robot._execute = Mock()
        robot.current_pose = Mock(return_value=np.eye(4))
        joints = JointState()
        joints.name = [f'fr3_joint{i}' for i in range(1, 8)]
        joints.position = [0.] * 7
        robot._robot_state = joints, time.monotonic()
        robot.gripper_state = Mock(return_value={'width_m': .020, 'joints': {'fr3_finger_joint1': .010, 'fr3_finger_joint2': .010}})
        robot._cartesian = Mock()
        robot._cartesian.wait_for_service.return_value = True
        response = GetCartesianPath.Response()
        response.error_code.val, response.fraction = 1, 1.
        response.solution.joint_trajectory.joint_names = list(joints.name)
        first, last = JointTrajectoryPoint(), JointTrajectoryPoint()
        first.positions, last.positions = [0.] * 7, [.001] * 7
        last.time_from_start.sec = 1
        response.solution.joint_trajectory.points = [first, last]
        robot._wait = Mock(return_value=response)
        robot._action = Mock(return_value=types.SimpleNamespace(error_code=MoveItErrorCodes(val=1)))
        target = np.eye(4)
        target[2, 3] = .02
        return robot, target

    def test_cartesian_seed_replaces_stale_zero_fingers_without_mutating_arm_cache(self):
        robot,target=self._planned_fixture()
        cached=robot._robot_state[0]
        cached.name.extend(['fr3_finger_joint1','fr3_finger_joint2'])
        cached.position.extend([0.,0.])
        robot.plan_linear(target)
        seed=robot._cartesian.call_async.call_args.args[0].start_state.joint_state
        self.assertEqual(list(seed.position[-2:]),[.010,.010])
        self.assertEqual(list(cached.position[-2:]),[0.,0.])
        self.assertEqual(len(seed.name),9)
        self.assertEqual(list(seed.velocity),[])
        self.assertEqual(list(seed.effort),[])

    def test_lost_read_only_planning_response_retries_once_without_execution(self):
        robot,target=self._planned_fixture()
        response=robot._wait.return_value
        first,second=Mock(),Mock()
        robot._cartesian.call_async.side_effect=[first,second]
        robot._wait.side_effect=[ROSResponseTimeout('lost response'),response]
        plan=robot.plan_linear(target)
        self.assertIsInstance(plan,LinearPlan)
        self.assertEqual(robot._cartesian.call_async.call_count,2)
        robot._cartesian.remove_pending_request.assert_called_once_with(first)
        first.cancel.assert_called_once()
        self.assertEqual([c.args[1]for c in robot._wait.call_args_list],[8.,8.])
        robot._action.assert_not_called()

    def test_planning_response_retry_is_strictly_bounded(self):
        robot,target=self._planned_fixture()
        robot._wait.side_effect=ROSResponseTimeout('lost response')
        with self.assertRaisesRegex(MotionError,'2 bounded attempts'):
            robot.plan_linear(target)
        self.assertEqual(robot._cartesian.call_async.call_count,2)
        self.assertEqual(robot._cartesian.remove_pending_request.call_count,2)
        robot._action.assert_not_called()

    def test_planning_retry_refuses_changed_robot_start(self):
        robot,target=self._planned_fixture()
        moved=np.eye(4);moved[0,3]=.002
        robot.current_pose.side_effect=[np.eye(4),moved]
        robot._wait.side_effect=ROSResponseTimeout('lost response')
        with self.assertRaisesRegex(MotionError,'moved during a lost planning response'):
            robot.plan_linear(target)
        self.assertEqual(robot._cartesian.call_async.call_count,1)
        robot._action.assert_not_called()

    def test_cartesian_discovery_settle_occurs_once_per_adapter(self):
        robot,target=self._planned_fixture()
        robot.cfg['planning_service_discovery_settle_sec']=.35
        with patch('lego_sorter.motion.time.sleep')as sleep:
            robot.plan_linear(target)
            robot.plan_linear(target)
        sleep.assert_called_once_with(.35)
        robot._action.assert_not_called()

    def test_execution_timeout_is_cancelled_never_retried(self):
        robot=adapter();client=Mock();client.wait_for_server.return_value=True
        handle=Mock();handle.accepted=True
        robot._wait=Mock(side_effect=[handle,ROSResponseTimeout('execution timeout'),object()])
        with self.assertRaises(ROSResponseTimeout):robot._action(client,object())
        client.send_goal_async.assert_called_once()
        handle.cancel_goal_async.assert_called_once()

    def test_plan_is_reviewable_and_does_not_execute(self):
        robot, target = self._planned_fixture()
        plan = robot.plan_linear(target)
        self.assertIsInstance(plan, LinearPlan)
        robot._action.assert_not_called()
        review = plan.to_dict()
        self.assertFalse(review['noop'])
        self.assertEqual(review['duration_sec'], 1.)
        self.assertEqual(review['joint_ranges_rad']['fr3_joint4'], [0., .001])
        self.assertAlmostEqual(review['T_base_tcp_target'][2][3], .02)
        robot.current_pose.side_effect = [np.eye(4), target]
        np.testing.assert_allclose(robot.execute_linear_plan(plan), target)
        robot._action.assert_called_once()

    def test_explicit_physical_goal_is_verified_after_corrected_command(self):
        robot, command = self._planned_fixture()
        plan = robot.plan_linear(command)
        physical = command.copy()
        physical[2, 3] -= .002
        robot.current_pose.side_effect = [np.eye(4), physical]
        np.testing.assert_allclose(robot.execute_linear_plan(plan, verification_target=physical), physical)
        robot._action.assert_called_once()

    def test_distant_physical_verification_goal_refused_before_action(self):
        robot, command = self._planned_fixture()
        plan = robot.plan_linear(command)
        physical = command.copy()
        physical[2, 3] -= .004
        with self.assertRaisesRegex(MotionError, 'verification target exceeds'):
            robot.execute_linear_plan(plan, verification_target=physical)
        robot._action.assert_not_called()

    def test_nonrigid_physical_verification_goal_refused_before_action(self):
        robot, command = self._planned_fixture()
        plan = robot.plan_linear(command)
        physical = command.copy()
        physical[0, 0] = 2.
        with self.assertRaises(ValueError):
            robot.execute_linear_plan(plan, verification_target=physical)
        robot._action.assert_not_called()

    def test_reviewed_plan_refuses_robot_pose_drift(self):
        robot, target = self._planned_fixture()
        plan = robot.plan_linear(target)
        changed = np.eye(4)
        changed[0, 3] = .002
        robot.current_pose.return_value = changed
        with self.assertRaisesRegex(MotionError, 'moved since Cartesian planning'):
            robot.execute_linear_plan(plan)
        robot._action.assert_not_called()

    def test_reviewed_plan_refuses_arm_joint_drift(self):
        robot, target = self._planned_fixture()
        plan = robot.plan_linear(target)
        robot._robot_state[0].position[3] = .03
        with self.assertRaisesRegex(MotionError, 'Arm joints changed'):
            robot.execute_linear_plan(plan)
        robot._action.assert_not_called()

    def test_zero_distance_linear_returns_measured_pose_without_planning(self):
        robot = adapter()
        actual = np.eye(4)
        actual[:3, 3] = [.4, .1, .3]
        robot.current_pose = Mock(return_value=actual)
        robot._cartesian, robot._action = Mock(), Mock()
        returned = robot.move_linear(actual.copy())
        self.assertIs(returned, actual)
        robot._cartesian.call_async.assert_not_called()
        robot._action.assert_not_called()

    def test_tiny_linear_noise_is_noop_but_not_placement_tolerance(self):
        robot = adapter()
        actual = np.eye(4)
        robot.current_pose = Mock(return_value=actual)
        robot._cartesian, robot._action = Mock(), Mock()
        target = actual.copy()
        target[2, 3] += .00002  # 20 microns: state quantization/noise scale.
        np.testing.assert_allclose(robot.move_linear(target), actual)
        target[2, 3] = .0005  # 0.5 mm is meaningful despite <3 mm pose tolerance.
        with self.assertRaisesRegex(MotionError, 'fresh robot joint state'):
            robot.move_linear(target)
        robot._action.assert_not_called()

    def test_linear_same_position_with_rotation_is_not_noop(self):
        robot = adapter()
        robot.current_pose = Mock(return_value=np.eye(4))
        robot._cartesian, robot._action = Mock(), Mock()
        angle = .01
        target = np.eye(4)
        target[:2, :2] = [[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]]
        with self.assertRaisesRegex(MotionError, 'fresh robot joint state'):
            robot.move_linear(target)
        robot._action.assert_not_called()

    def test_noop_still_requires_fresh_measured_tf(self):
        robot = adapter()
        robot.current_pose = Mock(side_effect=MotionError('Robot TCP transform is stale'))
        with self.assertRaisesRegex(MotionError, 'stale'):
            robot.move_linear(np.eye(4))

    def test_gripper_width_uses_sum_of_fingers_and_key(self):
        robot = adapter()
        msg = JointState()
        msg.name = ['fr3_finger_joint1', 'fr3_finger_joint2']
        msg.position = [.008, .008]
        robot._on_state(msg)
        state = robot.gripper_state()
        self.assertAlmostEqual(state['width_m'], .016)
        self.assertIn('joints', state)

    def test_narrow_preopen_uses_selected_width(self):
        try:
            from franka_msgs.action import Move
        except ImportError:
            self.skipTest('Source built Franka workspace for Move message test')
        robot=adapter()
        robot._Move,robot._open=Move,Mock()
        robot._action=Mock(return_value=types.SimpleNamespace(success=True))
        robot._wait_width=Mock(return_value=True)
        robot.open_gripper(width_m=.0198)
        self.assertAlmostEqual(robot._action.call_args.args[1].width,.0198)
        self.assertLess(robot._action.call_args.args[1].width,.06)

    def test_invalid_preopen_width_sends_no_command(self):
        robot=adapter();robot._action=Mock()
        for width in (0,-.01,float('nan'),float('inf'),.081):
            with self.subTest(width=width),self.assertRaises(ValueError):
                robot.open_gripper(width_m=width)
        robot._action.assert_not_called()

    def test_grasp_candidate_width_and_default(self):
        try:
            from franka_msgs.action import Grasp
        except ImportError:
            self.skipTest('Source built Franka workspace for grasp message test')
        robot = adapter()
        robot._Grasp, robot._grasp = Grasp, Mock()
        robot._action = Mock(return_value=types.SimpleNamespace(success=True))
        robot._wait_width = Mock(return_value=True)
        self.assertTrue(robot.grasp())
        first = robot._action.call_args.args[1]
        self.assertAlmostEqual(first.width, robot.cfg['grasp_width_m'])
        self.assertTrue(robot.grasp(width_m=.0318))
        long_grasp = robot._action.call_args.args[1]
        self.assertAlmostEqual(long_grasp.width, .0318)
        self.assertAlmostEqual(long_grasp.epsilon.inner, robot.cfg['grasp_epsilon_inner_m'])
        self.assertAlmostEqual(long_grasp.epsilon.outer, robot.cfg['grasp_epsilon_outer_m'])

    def test_invalid_candidate_grasp_width_sends_no_command(self):
        robot = adapter()
        robot._action = Mock()
        for width in (0, -.01, float('nan'), float('inf'), .08, .075):
            with self.subTest(width=width), self.assertRaises(ValueError):
                robot.grasp(width_m=width)
        robot._action.assert_not_called()

    def test_tf_translation_and_quaternion(self):
        msg = TransformStamped()
        msg.transform.rotation.x = 1.
        msg.transform.rotation.w = 0.
        msg.transform.translation.z = .2
        value = _from_tf(msg)
        np.testing.assert_allclose(value[:3, :3], np.diag([1., -1., -1.]))
        self.assertAlmostEqual(value[2, 3], .2)

    def test_interrupt_cancels_accepted_goal(self):
        robot = adapter()
        client = Mock()
        client.wait_for_server.return_value = True
        handle = Mock()
        handle.accepted = True
        robot._wait = Mock(side_effect=[handle, KeyboardInterrupt(), object()])
        with self.assertRaises(KeyboardInterrupt):
            robot._action(client, object())
        handle.cancel_goal_async.assert_called_once()

    def test_preflight_never_sends_goals(self):
        robot = adapter()
        for attr in ('_move', '_execute', '_open', '_grasp'):
            client = Mock()
            client.wait_for_server.return_value = True
            setattr(robot, attr, client)
        robot._cartesian = Mock()
        robot._cartesian.wait_for_service.return_value = True
        robot.current_pose = Mock(return_value=np.eye(4))
        robot.gripper_state = Mock(return_value={'width_m': .075})
        joints = JointState()
        joints.name = [f'fr3_joint{i}' for i in range(1, 8)]
        robot._robot_state = joints, time.monotonic()
        self.assertTrue(robot.preflight()['ready'])
        for client in (robot._move, robot._execute, robot._open, robot._grasp):
            client.send_goal_async.assert_not_called()


if __name__ == '__main__':
    unittest.main()
