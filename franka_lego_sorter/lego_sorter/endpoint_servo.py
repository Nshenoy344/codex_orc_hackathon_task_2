"""One bounded free-space endpoint correction using measured TCP feedback.

This is an explicit commissioning maneuver, not a contact servo or automatic
motion retry. Review the new collision-checked command plan before execution.
The commanded endpoint and the original physical goal remain separately logged.
No payload, controller gains, collision settings, or pose tolerances are changed.
"""
from dataclasses import dataclass
import math
import numpy as np
from .geometry import matrix


class EndpointServoError(RuntimeError):
    pass


def pose_error(goal, measured):
    goal, measured = matrix(goal), matrix(measured)
    translation = goal[:3, 3] - measured[:3, 3]
    angle = math.acos(float(np.clip(
        (np.trace(goal[:3, :3] @ measured[:3, :3].T)-1)/2, -1, 1)))
    return translation, angle


@dataclass
class EndpointCorrection:
    linear_plan: object
    physical_goal: np.ndarray
    measured_before: np.ndarray
    command_target: np.ndarray
    max_position_error_m: float
    max_rotation_error_rad: float

    def to_dict(self):
        delta, angle = pose_error(self.physical_goal, self.measured_before)
        return dict(kind='one_bounded_measured_endpoint_correction',
                    physical_goal=self.physical_goal.tolist(),
                    measured_before=self.measured_before.tolist(),
                    command_target=self.command_target.tolist(),
                    correction_translation_m=delta.tolist(), correction_rotation_rad=angle,
                    strict_position_tolerance_m=self.max_position_error_m,
                    strict_rotation_tolerance_rad=self.max_rotation_error_rad,
                    linear_plan=self.linear_plan.to_dict())


def plan_endpoint_correction(robot, physical_goal, limits=None):
    """Plan once; returns None if already converged. Sends no execution goal.

    Default use is the observed free-space upward lift checkpoint. A correction
    exceeding3mm/.01rad is refused. The new command moves at most6mm/.02rad
    from the measured pose, while final acceptance remains the original goal.
    """
    limits = dict(limits or {})
    goal, measured = matrix(physical_goal).copy(), matrix(robot.current_pose()).copy()
    tolerance = float(limits.get('position_tolerance_m', .001))
    rotation_tolerance = float(limits.get('rotation_tolerance_rad', robot.cfg['rotation_tolerance_rad']))
    if (not math.isfinite(tolerance) or tolerance <= 0 or
            tolerance > float(robot.cfg['position_tolerance_m'])):
        raise ValueError('Correction tolerance must be positive and no looser than normal pose tolerance')
    if (not math.isfinite(rotation_tolerance) or rotation_tolerance <= 0 or
            rotation_tolerance > float(robot.cfg['rotation_tolerance_rad'])):
        raise ValueError('Rotation tolerance must not loosen normal pose verification')
    delta, angle = pose_error(goal, measured)
    if np.linalg.norm(delta) <= tolerance and angle <= rotation_tolerance:
        return None
    # Hard ceilings agree with the motion adapter's explicit physical-goal bound.
    max_delta = float(limits.get('max_correction_m', .003))
    max_angle = float(limits.get('max_correction_rad', .01))
    if not 0 < max_delta <= .003 or not 0 < max_angle <= .01:
        raise ValueError('Correction ceilings cannot exceed3mm/.01rad')
    if np.linalg.norm(delta) > max_delta or angle > max_angle:
        raise EndpointServoError(f'Measured endpoint error is outside correction bound: {np.linalg.norm(delta):.6f}m/{angle:.6f}rad')
    if limits.get('require_upward_translation', True) and delta[2] < -.0001:
        raise EndpointServoError('This free-space lift correction would command a downward endpoint offset')
    command = goal.copy()
    command[:3, 3] += delta
    command[:3, :3] = (goal[:3, :3] @ measured[:3, :3].T) @ goal[:3, :3]
    travel, travel_angle = pose_error(command, measured)
    if np.linalg.norm(travel) > .006 + 1e-12 or travel_angle > .02 + 1e-12:
        raise EndpointServoError('Correction command exceeds6mm/.02rad measured-pose envelope')
    plan = robot.plan_linear(command)
    if plan.noop:
        raise EndpointServoError('Nonconverged correction unexpectedly produced a no-op plan')
    return EndpointCorrection(plan, goal, measured, command, tolerance, rotation_tolerance)


def execute_endpoint_correction(robot, correction):
    """Execute exactly one already-reviewed correction, with no retry loop."""
    if not isinstance(correction, EndpointCorrection):
        raise TypeError('Expected an EndpointCorrection returned by plan_endpoint_correction')
    actual = matrix(robot.execute_linear_plan(correction.linear_plan,
                                               verification_target=correction.physical_goal))
    delta, angle = pose_error(correction.physical_goal, actual)
    report = correction.to_dict()
    report.update(measured_after=actual.tolist(), residual_translation_m=delta.tolist(),
                  residual_position_error_m=float(np.linalg.norm(delta)), residual_rotation_error_rad=angle,
                  success=bool(np.linalg.norm(delta) <= correction.max_position_error_m and
                               angle <= correction.max_rotation_error_rad), attempts=1)
    if not report['success']:
        raise EndpointServoError(f'One correction did not converge; stop for inspection: {report}')
    return report
