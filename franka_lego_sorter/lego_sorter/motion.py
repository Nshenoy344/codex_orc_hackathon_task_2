"""ROS 2 Humble MoveIt / Franka Hand adapter (no motion on construction).

Configuration is a ``motion`` mapping (or the mapping itself). All Cartesian
poses are T_base_tcp: homogeneous matrices in meters. Required deployment
services: MoveGroup at /move_action, Franka Move and Grasp actions, base-to-TCP
TF, and gripper JointState. Defaults match the installed FR3 Humble workspace.
Uses its own rclpy Context, executor thread, TF subscriptions, and action clients;
callers must not spin this node. All commands wait for terminal results.
"""
from __future__ import annotations

import math
import copy
import threading
import time
from typing import Optional, Any
from dataclasses import dataclass

import numpy as np
from .application_joint_bounds import validate_application_joint_bounds


class MotionError(RuntimeError):
    """A planning, execution, verification, or communication failure."""


class ROSResponseTimeout(MotionError):
    """A local wait expired; unlike an action, a planning query may be retried."""


def _matrix(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError('Pose must be a finite 4x4 transform')
    if not np.allclose(value[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError('Invalid homogeneous transform bottom row')
    rot = value[:3, :3]
    if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rot), 1, atol=1e-5):
        raise ValueError('Pose rotation must belong to SO(3)')
    return value


def _quaternion(rot):
    # Eigenvector form is stable near pi, unlike trace-only conversions.
    k = np.array([
        [rot[0, 0]-rot[1, 1]-rot[2, 2], rot[0, 1]+rot[1, 0], rot[0, 2]+rot[2, 0], rot[2, 1]-rot[1, 2]],
        [rot[0, 1]+rot[1, 0], rot[1, 1]-rot[0, 0]-rot[2, 2], rot[1, 2]+rot[2, 1], rot[0, 2]-rot[2, 0]],
        [rot[0, 2]+rot[2, 0], rot[1, 2]+rot[2, 1], rot[2, 2]-rot[0, 0]-rot[1, 1], rot[1, 0]-rot[0, 1]],
        [rot[2, 1]-rot[1, 2], rot[0, 2]-rot[2, 0], rot[1, 0]-rot[0, 1], rot.trace()]
    ]) / 3.0
    _, vectors = np.linalg.eigh(k)
    q = vectors[:, -1]
    return q if q[3] >= 0 else -q


def _from_tf(tf):
    p, q = tf.transform.translation, tf.transform.rotation
    x, y, z, w = np.array([q.x, q.y, q.z, q.w]) / np.linalg.norm([q.x, q.y, q.z, q.w])
    out = np.eye(4)
    out[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                   [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                   [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    out[:3, 3] = [p.x, p.y, p.z]
    return _matrix(out)


@dataclass(frozen=True)
class LinearPlan:
    """Reviewable Cartesian plan; creating one never executes a trajectory."""
    start_pose: np.ndarray
    target_pose: np.ndarray
    start_joint_names: tuple
    start_joint_positions: tuple
    trajectory: Any = None
    fraction: float = 1.0

    @property
    def noop(self):
        return self.trajectory is None

    def to_dict(self):
        data = dict(noop=self.noop, fraction=self.fraction,
                    T_base_tcp_start=self.start_pose.tolist(),
                    T_base_tcp_target=self.target_pose.tolist(),
                    start_joint_names=list(self.start_joint_names),
                    start_joint_positions=list(self.start_joint_positions))
        if not self.noop:
            jt = self.trajectory.joint_trajectory
            data['joint_names'] = list(jt.joint_names)
            data['points'] = [dict(positions=list(p.positions), velocities=list(p.velocities),
                                   accelerations=list(p.accelerations),
                                   time_sec=p.time_from_start.sec + p.time_from_start.nanosec/1e9)
                              for p in jt.points]
            data['duration_sec'] = data['points'][-1]['time_sec']
            positions = np.array([p.positions for p in jt.points])
            data['joint_ranges_rad'] = {name: [float(positions[:, i].min()), float(positions[:, i].max())]
                                       for i, name in enumerate(jt.joint_names)}
        return data


class RosRobot:
    """Blocking command adapter; invoke from one caller thread only.

    Supported config keys are declared in DEFAULTS. ``duration`` in move_pose
    sets a minimum planning-time allowance, not an imposed trajectory duration;
    MoveIt time-parameterizes the path using velocity/acceleration scaling.
    """
    DEFAULTS = dict(
        base_frame='fr3_link0', tcp_frame='fr3_hand_tcp', planning_group='fr3_arm',
        move_group_action='/move_action', cartesian_path_service='/compute_cartesian_path',
        execute_trajectory_action='/execute_trajectory', robot_joint_states_topic='/joint_states',
        cartesian_step_m=.003, cartesian_jump_threshold=2.0,
        cartesian_revolute_jump_rad=.3, cartesian_prismatic_jump_m=.02,
        gripper_move_action='/franka_gripper/move',
        gripper_grasp_action='/franka_gripper/grasp',
        joint_states_topic='/franka_gripper/joint_states',
        finger_joint_names=['fr3_finger_joint1', 'fr3_finger_joint2'],
        velocity_scaling=0.08, acceleration_scaling=0.08,
        open_width_m=0.075, grasp_width_m=0.0158, grasp_force_n=15.0,
        gripper_speed_mps=0.03, grasp_epsilon_inner_m=0.003,
        grasp_epsilon_outer_m=0.003, action_timeout_sec=60.0,
        state_timeout_sec=2.0, position_tolerance_m=0.003,
        planning_service_timeout_sec=8.0, planning_service_retries=1,
        planning_service_discovery_settle_sec=0.35,
        rotation_tolerance_rad=0.04, planning_time_sec=5.0,
        gripper_width_tolerance_m=0.004)

    def __init__(self, config):
        import rclpy
        from rclpy.context import Context
        from rclpy.node import Node
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.action import ActionClient
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import JointState
        from tf2_ros import Buffer, TransformListener
        from moveit_msgs.action import MoveGroup, ExecuteTrajectory
        from moveit_msgs.srv import GetCartesianPath
        from franka_msgs.action import Move, Grasp
        self.cfg = dict(self.DEFAULTS)
        self.cfg.update(config.get('motion', config))
        for name in ('velocity_scaling', 'acceleration_scaling'):
            if not 0 < float(self.cfg[name]) <= 1:
                raise ValueError(f'{name} must lie in (0, 1]')
        for name in ('action_timeout_sec', 'state_timeout_sec', 'planning_time_sec',
                     'gripper_speed_mps', 'position_tolerance_m', 'rotation_tolerance_rad',
                     'gripper_width_tolerance_m', 'grasp_force_n', 'planning_service_timeout_sec'):
            if not math.isfinite(float(self.cfg[name])) or float(self.cfg[name]) <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if self.cfg['planning_service_timeout_sec']>30:
            raise ValueError('Read-only planning response timeout must be<=30seconds')
        retries=self.cfg['planning_service_retries']
        if isinstance(retries,bool) or not isinstance(retries,int) or not 0<=retries<=1:
            raise ValueError('Read-only planning retries must be zero or one')
        settle=float(self.cfg['planning_service_discovery_settle_sec'])
        if not math.isfinite(settle) or not 0<=settle<=2:
            raise ValueError('Planning discovery settle must be finite in[0,2]seconds')
        if not 0 < self.cfg['grasp_width_m'] < self.cfg['open_width_m'] <= .08:
            raise ValueError('Need 0 < grasp_width_m < open_width_m <= 0.08')
        self._rclpy = rclpy
        self._context = Context()
        rclpy.init(context=self._context)
        self._node = Node('lego_sorter_motion', context=self._context)
        self._executor = MultiThreadedExecutor(num_threads=2, context=self._context)
        self._executor.add_node(self._node)
        self._buffer = Buffer()
        self._listener = TransformListener(self._buffer, self._node)
        self._MoveGroup, self._Move, self._Grasp = MoveGroup, Move, Grasp
        self._GetCartesianPath, self._ExecuteTrajectory = GetCartesianPath, ExecuteTrajectory
        self._cartesian = self._node.create_client(GetCartesianPath, self.cfg["cartesian_path_service"])
        self._execute = ActionClient(self._node, ExecuteTrajectory, self.cfg["execute_trajectory_action"])
        self._move = ActionClient(self._node, MoveGroup, self.cfg['move_group_action'])
        self._open = ActionClient(self._node, Move, self.cfg['gripper_move_action'])
        self._grasp = ActionClient(self._node, Grasp, self.cfg['gripper_grasp_action'])
        self._state_lock = threading.Lock()
        self._state = None
        self._robot_state = None
        self._robot_subscription = self._node.create_subscription(
            JointState, self.cfg["robot_joint_states_topic"], self._on_robot_state, qos_profile_sensor_data)
        self._subscription = self._node.create_subscription(
            JointState, self.cfg['joint_states_topic'], self._on_state, qos_profile_sensor_data)
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._closed = False
        self._cartesian_discovery_settled = False
        self._thread.start()

    def _spin(self):
        from rclpy.executors import ExternalShutdownException
        try:
            self._executor.spin()
        except ExternalShutdownException:
            pass

    def _on_robot_state(self, msg):
        with self._state_lock:
            self._robot_state = (msg, time.monotonic())

    def _on_state(self, msg):
        with self._state_lock:
            self._state = (msg, time.monotonic())

    def _wait(self, future, timeout):
        event = threading.Event()
        future.add_done_callback(lambda _: event.set())
        if not event.wait(timeout):
            raise ROSResponseTimeout(f'ROS response timeout after {timeout:.1f}s')
        error = future.exception()
        if error:
            raise MotionError(str(error)) from error
        return future.result()


    def _cartesian_response(self, request, expected_start):
        """Bounded retry of a READ-ONLY plan query after lost service responses.

        FastDDS can discover the request server before the new client's reply
        reader is fully matched. Its wait_for_service readiness alone does not
        establish bidirectional response delivery. Let the executor/discovery
        thread settle once; a timeout then permits at most one identical planning
        query retry, never an action/execution retry. Late futures are detached.
        """
        timeout=float(self.cfg['planning_service_timeout_sec'])
        retries=int(self.cfg['planning_service_retries'])
        settle=float(self.cfg['planning_service_discovery_settle_sec'])
        if not getattr(self,'_cartesian_discovery_settled',False):
            if settle:time.sleep(settle)
            self._cartesian_discovery_settled=True
        for attempt in range(retries+1):
            if attempt:
                current=self.current_pose()
                moved=float(np.linalg.norm(current[:3,3]-expected_start[:3,3]))
                rotated=math.acos(float(np.clip(
                    (np.trace(current[:3,:3].T@expected_start[:3,:3])-1)/2,-1,1)))
                if moved>.001 or rotated>.01:
                    raise MotionError('Robot moved during a lost planning response; capture and replan')
                if not self._cartesian.wait_for_service(timeout_sec=self.cfg['state_timeout_sec']):
                    raise MotionError('Cartesian planning service vanished before read-only retry')
                if settle:time.sleep(settle)
            future=self._cartesian.call_async(request)
            try:
                return self._wait(future,timeout)
            except ROSResponseTimeout as error:
                self._cartesian.remove_pending_request(future)
                future.cancel()
                if attempt==retries:
                    raise MotionError(f'Cartesian planning response unavailable after {retries+1} bounded attempts; no execution sent') from error
                node=getattr(self,'_node',None)
                if node is not None:
                    node.get_logger().warning('Read-only Cartesian planning response timed out; retrying once after discovery settle')
            except BaseException:
                self._cartesian.remove_pending_request(future)
                future.cancel()
                raise


    def _action(self, client, goal):
        from action_msgs.msg import GoalStatus
        timeout = float(self.cfg['action_timeout_sec'])
        if self._closed:
            raise MotionError('Robot adapter is closed')
        if not client.wait_for_server(timeout_sec=self.cfg['state_timeout_sec']):
            raise MotionError(f'Action server unavailable: {client._action_name}')
        goal_future = client.send_goal_async(goal)
        try:
            handle = self._wait(goal_future, timeout)
        except BaseException:
            # A delayed acceptance must not leave a command running after timeout.
            def cancel_late(future):
                try:
                    late = future.result()
                    if late and late.accepted:
                        late.cancel_goal_async()
                except Exception:
                    pass
            goal_future.add_done_callback(cancel_late)
            raise
        if not handle.accepted:
            raise MotionError('Action goal was rejected')
        try:
            result = self._wait(handle.get_result_async(), timeout)
        except BaseException:
            try:
                self._wait(handle.cancel_goal_async(), min(3., timeout))
            except MotionError:
                pass
            raise
        if result.status != GoalStatus.STATUS_SUCCEEDED:
            raise MotionError(f'Action terminal status {result.status}')
        return result.result

    def preflight(self):
        """Read-only readiness report. Never sends action goals or moves the hand."""
        timeout = float(self.cfg['state_timeout_sec'])
        actions = {
            'move_group': self._move.wait_for_server(timeout_sec=timeout),
            'execute_trajectory': self._execute.wait_for_server(timeout_sec=timeout),
            'gripper_move': self._open.wait_for_server(timeout_sec=timeout),
            'gripper_grasp': self._grasp.wait_for_server(timeout_sec=timeout),
        }
        report = dict(actions=actions,
                      cartesian_path=self._cartesian.wait_for_service(timeout_sec=timeout))
        try:
            report['T_base_tcp'] = self.current_pose().tolist()
        except MotionError as error:
            report['pose_error'] = str(error)
        try:
            report['gripper'] = self.gripper_state()
        except MotionError as error:
            report['gripper_error'] = str(error)
        with self._state_lock:
            snapshot = self._robot_state
        required = set(self.cfg.get('arm_joint_names', [f'fr3_joint{i}' for i in range(1, 8)]))
        report['fresh_arm_joints'] = bool(snapshot and
            time.monotonic() - snapshot[1] <= timeout and required.issubset(snapshot[0].name))
        report['ready'] = (all(actions.values()) and report['cartesian_path'] and
                           report['fresh_arm_joints'] and 'T_base_tcp' in report and 'gripper' in report)
        return report

    def current_pose(self):
        from rclpy.time import Time
        from rclpy.duration import Duration
        timeout = float(self.cfg['state_timeout_sec'])
        try:
            tf = self._buffer.lookup_transform(self.cfg['base_frame'], self.cfg['tcp_frame'],
                                               Time(), timeout=Duration(seconds=timeout))
        except Exception as error:
            raise MotionError(f'Base-to-TCP TF unavailable: {error}') from error
        stamp = Time.from_msg(tf.header.stamp)
        # A zero stamp may represent an entirely static chain; reject for robot TCP.
        age = (self._node.get_clock().now() - stamp).nanoseconds / 1e9
        if not 0 <= age <= timeout:
            raise MotionError(f'Robot TCP transform is stale/future ({age:.3f}s)')
        return _from_tf(tf)

    def move_pose(self, T, duration: Optional[float] = None):
        from geometry_msgs.msg import Pose
        from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, JointConstraint
        from shape_msgs.msg import SolidPrimitive
        target = _matrix(T)
        goal = self._MoveGroup.Goal()
        req = goal.request
        req.group_name = self.cfg['planning_group']
        req.num_planning_attempts = 5
        req.allowed_planning_time = max(float(self.cfg['planning_time_sec']), float(duration or 0))
        req.max_velocity_scaling_factor = float(self.cfg['velocity_scaling'])
        req.max_acceleration_scaling_factor = float(self.cfg['acceleration_scaling'])
        req.start_state.is_diff = True
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = map(float, target[:3, 3])
        q = _quaternion(target[:3, :3])
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(float, q)
        pc = PositionConstraint()
        pc.header.frame_id = self.cfg['base_frame']
        pc.link_name = self.cfg['tcp_frame']
        pc.weight = 1.0
        shape = SolidPrimitive()
        shape.type = SolidPrimitive.SPHERE
        shape.dimensions = [float(self.cfg['position_tolerance_m'])]
        pc.constraint_region.primitives = [shape]
        pc.constraint_region.primitive_poses = [pose]
        oc = OrientationConstraint()
        oc.header.frame_id = self.cfg['base_frame']
        oc.link_name = self.cfg['tcp_frame']
        oc.orientation = pose.orientation
        oc.absolute_x_axis_tolerance = float(self.cfg['rotation_tolerance_rad'])
        oc.absolute_y_axis_tolerance = float(self.cfg['rotation_tolerance_rad'])
        oc.absolute_z_axis_tolerance = float(self.cfg['rotation_tolerance_rad'])
        oc.weight = 1.0
        constraints = Constraints()
        constraints.position_constraints = [pc]
        constraints.orientation_constraints = [oc]
        req.goal_constraints = [constraints]
        # Application branch limits after the observed internal self-avoidance reflex.
        # These constrain planning only; no Franka threshold is modified.
        for name, center in [('fr3_joint5', 0.), ('fr3_joint6', 1.5)]:
            bound = JointConstraint()
            bound.joint_name = name
            bound.position = center
            bound.tolerance_above = 1.5
            bound.tolerance_below = 1.5
            bound.weight = 1.0
            req.path_constraints.joint_constraints.append(bound)
        goal.planning_options.plan_only = False
        goal.planning_options.replan = False
        result = self._action(self._move, goal)
        if result.error_code.val != result.error_code.SUCCESS:
            raise MotionError(f'MoveIt error code: {result.error_code.val}')
        return self._verify_pose(target)

    def _verify_pose(self, target):
        # Allow the state publisher to catch up after the action completes.
        deadline = time.monotonic() + float(self.cfg['state_timeout_sec'])
        while True:
            actual = self.current_pose()
            distance = np.linalg.norm(actual[:3, 3] - target[:3, 3])
            angle = math.acos(float(np.clip((np.trace(actual[:3, :3].T @ target[:3, :3])-1)/2, -1, 1)))
            if distance <= self.cfg['position_tolerance_m'] and angle <= self.cfg['rotation_tolerance_rad']:
                return actual
            if time.monotonic() >= deadline:
                raise MotionError(f'Pose verification failed: {distance:.4f}m, {angle:.4f}rad')
            time.sleep(.05)

    def _fresh_arm_state(self):
        """Wait briefly for initial discovery; never return missing or stale state."""
        deadline = time.monotonic() + float(self.cfg['state_timeout_sec'])
        while True:
            with self._state_lock:
                snapshot = self._robot_state
            if snapshot is not None and 0 <= time.monotonic() - snapshot[1] <= self.cfg['state_timeout_sec']:
                return snapshot
            if time.monotonic() >= deadline:
                raise MotionError('No fresh robot joint state available before Cartesian planning/execution')
            time.sleep(.01)

    def plan_linear(self, T):
        """Plan a complete collision-checked straight TCP segment without execution.

        Requires fresh measured joint state. Rejects partial paths, un-timed
        trajectories, and jumps. The wrist camera's physical geometry must be
        present in the robot model for MoveIt collision checks to include it.
        """
        from geometry_msgs.msg import Pose
        target = _matrix(T)
        actual = self.current_pose()  # Verify TF is live even for a no-op.
        # The runner intentionally lifts to transit before each XY move, so an
        # already-raised TCP frequently receives its current target again. A
        # zero-displacement Cartesian path can be empty or have zero duration;
        # return the measured pose instead of sending that path to a controller.
        # These bounds (0.05 mm, 0.03 deg) are deliberately far tighter than the
        # placement tolerance and cannot skip a meaningful pick/place correction.
        delta_m = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
        delta_rad = math.acos(float(np.clip(
            (np.trace(actual[:3, :3].T @ target[:3, :3]) - 1) / 2, -1, 1)))
        if delta_m <= .00005 and delta_rad <= .0005:
            return LinearPlan(actual.copy(), target.copy(), (), ())
        snapshot = self._fresh_arm_state()
        joint_state = copy.deepcopy(snapshot[0])
        required = {f'fr3_joint{i}' for i in range(1, 8)}
        # Explicit names are configurable for namespaced/prefixed descriptions.
        required = set(self.cfg.get('arm_joint_names', required))
        if not required.issubset(joint_state.name):
            raise MotionError('Robot joint state lacks configured arm joints')
        if not self._cartesian.wait_for_service(timeout_sec=self.cfg['state_timeout_sec']):
            raise MotionError('MoveIt Cartesian path service unavailable')
        # MoveIt's arm JointState monitor can retain zero finger positions even
        # while the independent Hand publisher reports an open gripper. Seed the
        # complete measured state so collision checking uses the physical fingers.
        hand = self.gripper_state()
        seed_names = list(joint_state.name)
        seed_positions = list(joint_state.position)
        for name, position in hand['joints'].items():
            if name in seed_names:
                seed_positions[seed_names.index(name)] = float(position)
            else:
                seed_names.append(name)
                seed_positions.append(float(position))
        joint_state.name = seed_names
        joint_state.position = seed_positions
        joint_state.velocity = []
        joint_state.effort = []
        req = self._GetCartesianPath.Request()
        req.header.frame_id = self.cfg['base_frame']
        req.start_state.joint_state = joint_state
        req.start_state.is_diff = True
        req.group_name = self.cfg['planning_group']
        req.link_name = self.cfg['tcp_frame']
        req.max_step = float(self.cfg['cartesian_step_m'])
        if not .0005 <= req.max_step <= .01:
            raise ValueError('cartesian_step_m must be between 0.0005 and 0.01')
        req.jump_threshold = float(self.cfg['cartesian_jump_threshold'])
        req.revolute_jump_threshold = float(self.cfg['cartesian_revolute_jump_rad'])
        req.prismatic_jump_threshold = float(self.cfg['cartesian_prismatic_jump_m'])
        if min(req.jump_threshold, req.revolute_jump_threshold, req.prismatic_jump_threshold) <= 0:
            raise ValueError('Cartesian jump thresholds must be positive')
        req.avoid_collisions = True
        req.max_velocity_scaling_factor = float(self.cfg['velocity_scaling'])
        req.max_acceleration_scaling_factor = float(self.cfg['acceleration_scaling'])
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = map(float, target[:3, 3])
        q = _quaternion(target[:3, :3])
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(float, q)
        req.waypoints = [pose]
        response = self._cartesian_response(req, actual)
        if (response.error_code.val != response.error_code.SUCCESS or
                not np.isfinite(response.fraction) or response.fraction < 1.0 - 1e-6):
            raise MotionError(f'Incomplete Cartesian path: fraction={response.fraction}, error={response.error_code.val}')
        points = response.solution.joint_trajectory.points
        if not points:
            raise MotionError('Cartesian planner returned an empty trajectory')
        times = [p.time_from_start.sec + p.time_from_start.nanosec / 1e9 for p in points]
        if len(points) > 1 and (times[-1] <= 0 or any(b <= a for a, b in zip(times, times[1:]))):
            raise MotionError('Cartesian trajectory has invalid time parameterization')
        names = list(response.solution.joint_trajectory.joint_names)
        if not names or any(len(p.positions) != len(names) or not np.isfinite(p.positions).all() for p in points):
            raise MotionError('Cartesian trajectory has malformed joint positions')
        try:validate_application_joint_bounds(response.solution)
        except ValueError as exc:raise MotionError(str(exc)) from exc
        return LinearPlan(actual.copy(), target.copy(), tuple(joint_state.name),
                          tuple(joint_state.position), response.solution, float(response.fraction))

    def execute_linear_plan(self, plan, verification_target=None):
        """Execute a reviewed plan, then verify its physical endpoint.

        A bias-compensated command may name an explicit nearby physical goal.
        It must be a rigid transform within3mm/.01rad of the command target;
        this never changes measured-pose tolerances or action/result checks.
        """
        if not isinstance(plan, LinearPlan):
            raise TypeError('Expected a LinearPlan returned by plan_linear')
        actual = self.current_pose()
        start = _matrix(plan.start_pose)
        target = _matrix(plan.target_pose)
        physical_target = target if verification_target is None else _matrix(verification_target)
        if verification_target is not None:
            offset_m = float(np.linalg.norm(physical_target[:3, 3] - target[:3, 3]))
            offset_rad = math.acos(float(np.clip(
                (np.trace(physical_target[:3, :3].T @ target[:3, :3]) - 1) / 2, -1, 1)))
            if offset_m > .003 + 1e-12 or offset_rad > .01 + 1e-12:
                raise MotionError('Physical verification target exceeds3mm/.01rad command correction bound')
        distance = float(np.linalg.norm(actual[:3, 3] - start[:3, 3]))
        angle = math.acos(float(np.clip((np.trace(actual[:3, :3].T @ start[:3, :3])-1)/2, -1, 1)))
        if distance > .001 or angle > .01:
            raise MotionError(f'Robot moved since Cartesian planning: {distance:.5f}m, {angle:.5f}rad')
        if plan.noop:
            return actual if verification_target is None else self._verify_pose(physical_target)
        snapshot = self._fresh_arm_state()
        now = dict(zip(snapshot[0].name, snapshot[0].position))
        before = dict(zip(plan.start_joint_names, plan.start_joint_positions))
        required = set(plan.trajectory.joint_trajectory.joint_names)
        if (not required.issubset(now) or not required.issubset(before) or
                any(not np.isfinite(now[n]) or abs(now[n] - before[n]) > .01 for n in required)):
            raise MotionError('Arm joints changed since Cartesian planning; replan')
        try:validate_application_joint_bounds(plan.trajectory)
        except ValueError as exc:raise MotionError(str(exc)) from exc
        goal = self._ExecuteTrajectory.Goal()
        goal.trajectory = plan.trajectory
        result = self._action(self._execute, goal)
        if result.error_code.val != result.error_code.SUCCESS:
            raise MotionError(f'Cartesian execution failed: {result.error_code.val}')
        return self._verify_pose(physical_target)

    def move_linear(self, T):
        """Convenience plan-and-execute; use split methods to review before moving."""
        return self.execute_linear_plan(self.plan_linear(T))

    def gripper_state(self):
        deadline = time.monotonic() + float(self.cfg['state_timeout_sec'])
        while True:
            with self._state_lock:
                snapshot = self._state
            if snapshot is not None:
                msg, received = snapshot
                if time.monotonic() - received <= self.cfg['state_timeout_sec']:
                    joints = dict(zip(msg.name, msg.position))
                    names = self.cfg['finger_joint_names']
                    if all(name in joints for name in names):
                        width = float(sum(joints[name] for name in names))
                        if np.isfinite(width):
                            return dict(width_m=width, joints=joints,
                                        age_sec=time.monotonic() - received)
            if time.monotonic() >= deadline:
                raise MotionError('No fresh gripper JointState with configured finger joints')
            time.sleep(.03)

    def open_gripper(self, width_m=None):
        width=float(self.cfg['open_width_m'] if width_m is None else width_m)
        if not math.isfinite(width) or not 0 < width <= .08:
            raise ValueError('Move width must be finite and in (0,0.08] meters')
        goal = self._Move.Goal()
        goal.width, goal.speed = width, float(self.cfg['gripper_speed_mps'])
        result = self._action(self._open, goal)
        if not result.success:
            raise MotionError(f'Gripper open failed: {result.error}')
        if not self._wait_width(goal.width - self.cfg['gripper_width_tolerance_m'],
                                goal.width + self.cfg['gripper_width_tolerance_m']):
            raise MotionError('Gripper opening width verification failed')

    def grasp(self, width_m=None):
        """Grasp across a candidate's short (~16mm) or long (~32mm) sides.

        Overrides only target width; configured speed, force and measured-width
        success tolerances remain in effect. No command is sent for invalid widths.
        """
        width = float(self.cfg['grasp_width_m'] if width_m is None else width_m)
        if not math.isfinite(width) or not 0 < width < float(self.cfg['open_width_m']) or width >= .08:
            raise ValueError('Grasp width must be finite, positive, and below configured open width (<=0.08m)')
        goal = self._Grasp.Goal()
        goal.width = width
        goal.speed = float(self.cfg['gripper_speed_mps'])
        goal.force = float(self.cfg['grasp_force_n'])
        goal.epsilon.inner = float(self.cfg['grasp_epsilon_inner_m'])
        goal.epsilon.outer = float(self.cfg['grasp_epsilon_outer_m'])
        result = self._action(self._grasp, goal)
        if not result.success:
            return False
        return self._wait_width(goal.width - goal.epsilon.inner, goal.width + goal.epsilon.outer)

    def _wait_width(self, lower, upper):
        # Wait for a post-result state publication; the cached message may still
        # show the previous opening when the action result first arrives.
        after = time.monotonic()
        deadline = after + float(self.cfg['state_timeout_sec'])
        while time.monotonic() < deadline:
            with self._state_lock:
                snapshot = self._state
            if snapshot is not None and snapshot[1] >= after:
                state = self.gripper_state()
                if lower <= state['width_m'] <= upper:
                    return True
            time.sleep(.03)
        return False

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(timeout_sec=3.)
        self._thread.join(timeout=3.)
        self._listener.unregister()
        for client in (self._move, self._execute, self._open, self._grasp):
            client.destroy()
        self._node.destroy_client(self._cartesian)
        self._node.destroy_node()
        self._context.try_shutdown()
