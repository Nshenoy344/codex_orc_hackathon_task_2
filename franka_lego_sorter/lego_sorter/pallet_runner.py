"""Measured-template pallet picking; only verified transfers advance occupancy.

Source removal is checked in the selected old footprint after reprojecting it
between measured camera poses. It does not presume a single source tower or
require another brick to remain after the last pick.
"""
import logging
import math
import time
import cv2
import numpy as np
from .geometry import matrix, above
from .occupancy import TemplateRegistry
from .pallet import inspect_pallet
from .planner import SortPlanner
from .runner import SortRunner

LOG = logging.getLogger(__name__)


def _unit(value):
    value = np.asarray(value, dtype=float)
    length = np.linalg.norm(value)
    if value.shape != (3,) or not np.isfinite(value).all() or length < 1e-8:
        raise ValueError('Invalid measured candidate axis')
    return value / length


def _rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])


def candidate_base(candidate, observation, tcp_color):
    camera = matrix(observation) @ matrix(tcp_color)
    center = (camera @ np.r_[candidate.point_camera_m, 1.])[:3]
    closing = _unit(camera[:3,:3] @ _unit(candidate.grasp_closing_axis_camera))
    if not np.isfinite(center).all() or np.linalg.norm(closing[:2]) < .9:
        raise ValueError('Candidate grasp is not on a near-horizontal brick face')
    return center, closing


def candidate_pick_pose(candidate, observation, tcp_color, reference, tcp_to_fingertip_m,
                        finger_insertion_below_surface_m, tcp_closing_axis=(0.,1.,0.)):
    center, closing = candidate_base(candidate, observation, tcp_color)
    target = matrix(reference).copy()
    if -target[2,2] < .95:
        raise ValueError('Physical fingertip extension requires a near-downward stock-Hand TCP orientation')
    reference_axis = target[:3,:3] @ _unit(tcp_closing_axis)
    if np.linalg.norm(reference_axis[:2]) < .9:
        raise ValueError('Taught pick orientation must have a horizontal finger-closing axis')
    delta = ((math.atan2(closing[1], closing[0]) -
              math.atan2(reference_axis[1], reference_axis[0]) + np.pi/2) % np.pi - np.pi/2)
    target[:3,:3] = _rz(delta) @ target[:3,:3]
    return tcp_at_surface(center, target, tcp_to_fingertip_m, finger_insertion_below_surface_m)


def grasp_tcp_surface_offset(execution):
    """Validate geometry and return the nominal perfectly-vertical TCP height offset.

    tcp_to_fingertip_m is the verified distal-tip extension below the ROS TCP;
    finger_insertion_below_surface_m is the intended physical finger insertion.
    A deprecated TCP-below-top depth alone cannot specify safe stock geometry.
    """
    tip = execution.get('tcp_to_fingertip_m')
    insertion = execution.get('finger_insertion_below_surface_m')
    for name, value in (('tcp_to_fingertip_m', tip),
                        ('finger_insertion_below_surface_m', insertion)):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value):
            raise ValueError(f'{name} must be explicitly measured, finite and positive; legacy grasp depth is insufficient')
    if not 0 < tip <= .05:
        raise ValueError('tcp_to_fingertip_m must be positive and physically plausible for the stock Hand')
    if not 0 < insertion < .0096:
        raise ValueError('Finger insertion must address only the selected brick body (<9.6mm)')
    return float(tip - insertion)


def tcp_at_surface(surface, orientation_pose, tip_extension, insertion):
    """Place the physical distal-tip point at surface minus base-Z insertion."""
    grasp_tcp_surface_offset(dict(tcp_to_fingertip_m=tip_extension,
                                 finger_insertion_below_surface_m=insertion))
    target = matrix(orientation_pose).copy()
    if -target[2,2] < .95:
        raise ValueError('Physical fingertip extension requires a near-downward stock-Hand TCP orientation')
    surface = np.asarray(surface, dtype=float)
    if surface.shape != (3,) or not np.isfinite(surface).all():
        raise ValueError('Measured brick surface must be a finite 3D point')
    target[:3,3] = surface - np.array([0.,0.,insertion]) - target[:3,:3] @ np.array([0.,0.,tip_extension])
    return target


def surface_from_tcp(pose, tip_extension, insertion):
    """Recover the brick top point represented by an actual measured TCP pose."""
    grasp_tcp_surface_offset(dict(tcp_to_fingertip_m=tip_extension,
                                 finger_insertion_below_surface_m=insertion))
    pose = matrix(pose)
    if -pose[2,2] < .95:
        raise ValueError('Physical fingertip extension requires a near-downward stock-Hand TCP orientation')
    return pose[:3,3] + pose[:3,:3] @ np.array([0.,0.,tip_extension]) + np.array([0.,0.,insertion])


def check_grasp_insertion(actual_tcp, observed_surface, tip_extension, insertion_max,
                          width, padlength, closing_thickness, minimum_penetration_m=.0002):
    """Bound all stock-Hand toe corners relative to the observed horizontal top.

    Local +Z points from TCP toward the distal tip. Each jaw occupies localY
    from +/-opening/2 outward by its physical closing-direction thickness, and
    localX spans the physical pad length. This checks measured endpoint geometry,
    not an uncertainty guarantee for the observed plane or physical finger CAD.
    It never adjusts the target or commands a retry.
    """
    pose = matrix(actual_tcp)
    surface = np.asarray(observed_surface, dtype=float)
    if surface.shape != (3,) or not np.isfinite(surface).all():
        raise ValueError('Observed horizontal source surface must be a finite3D point')
    values = (tip_extension, insertion_max, width, padlength, closing_thickness, minimum_penetration_m)
    if (any(isinstance(v, bool) or not isinstance(v, (int,float,np.integer,np.floating)) or
            not np.isfinite(v) for v in values) or tip_extension <= 0 or insertion_max <= 0 or
            not 0 < width <= .08 or padlength <= 0 or closing_thickness <= 0 or
            not 0 <= minimum_penetration_m < insertion_max):
        raise ValueError('Physical fingertip endpoint-check dimensions must be finite and positive')
    if -pose[2,2] < .95:
        raise RuntimeError('Measured tool is not near-downward for the stock fingertip check')
    corners = np.array([[x*padlength/2,side*(width/2+thickness),tip_extension]
                        for x in (-1,1) for side in (-1,1)
                        for thickness in (0.,closing_thickness)])
    measured = (pose[:3,:3] @ corners.T).T + pose[:3,3]
    penetrations = surface[2]-measured[:,2]
    shallowest,deepest = float(penetrations.min()),float(penetrations.max())
    if shallowest < minimum_penetration_m-1e-10 or deepest > insertion_max+1e-10:
        raise RuntimeError(f'Physical fingertip penetration unsafe/unverified: range '
                           f'{shallowest*1000:.3f}..{deepest*1000:.3f}mm; '
                           f'allowed {minimum_penetration_m*1000:.3f}..{insertion_max*1000:.3f}mm; '
                           'stop without closing or downward retry')
    return dict(minimum_penetration_m=shallowest,maximum_penetration_m=deepest,
                tip_corners_base_m=measured.tolist(), measured_opening_width_m=float(width),
                assumption='Observed top is horizontal; measurement uncertainty is not calibrated')


def verify_narrow_preopen(actual_width, body_width, commanded_width, settings,
                          tracking_tolerance_m=.0001):
    """Verify measured narrow opening before entering the inter-brick gaps.

    Action success and the generic gripper width tolerance are insufficient for
    a millimetre opening margin. No opening/arm correction is attempted here.
    """
    values = (actual_width, body_width, commanded_width, tracking_tolerance_m)
    if (any(isinstance(v,bool) or not isinstance(v,(int,float,np.integer,np.floating)) or
            not np.isfinite(v) for v in values) or not 0 < body_width < commanded_width <= .08 or
            tracking_tolerance_m <= 0 or tracking_tolerance_m > (commanded_width-body_width)/2):
        raise ValueError('Narrow-preopen measurements and tracking tolerance are invalid')
    if actual_width <= body_width:
        raise RuntimeError('Measured narrow preopen does not clear the physical brick; stop before descent')
    if abs(actual_width-commanded_width) > tracking_tolerance_m+1e-12:
        raise RuntimeError(f'Measured narrow preopen {actual_width*1000:.3f}mm differs from '
                           f'{commanded_width*1000:.3f}mm command by more than '
                           f'{tracking_tolerance_m*1000:.3f}mm; stop before descent')
    thickness = float(settings.get('finger_clearance_m',.016))
    if not np.isfinite(thickness) or thickness <= 0:
        raise ValueError('Measured physical finger clearance must be positive and finite')
    swept = thickness+(actual_width-body_width)/2
    gap = settings.get('measured_interbrick_gap_m')
    if gap is not None:
        if isinstance(gap,bool) or not np.isfinite(gap) or gap <= 0:
            raise ValueError('Measured interbrick gap must be positive and finite')
        if swept > gap+1e-12:
            raise RuntimeError('Measured narrow preopen consumes the physical interbrick gap; stop before descent')
    return dict(measured_opening_width_m=float(actual_width), commanded_opening_width_m=float(commanded_width),
                opening_tracking_error_m=float(actual_width-commanded_width), swept_thickness_m=float(swept),
                remaining_measured_gap_m=None if gap is None else float(gap-swept))


def selected_grasp_width(candidate, settings):
    """Keep measured shape evidence separate from an explicitly nominal command."""
    source = settings.get('grasp_width_source', 'measured')
    if source == 'measured':
        value = candidate.grasp_width_m
    elif source == 'configured_2x4_nominal':
        if candidate.grasp_orientation not in ('across_width', 'across_length'):
            raise ValueError('Unknown grasp orientation for a nominal2x4 width')
        value = settings.get('expected_width_m', .0158) if candidate.grasp_orientation == 'across_width' else settings.get('expected_length_m', .0318)
    else:
        raise ValueError('grasp_width_source must be measured or configured_2x4_nominal')
    if isinstance(value, bool) or not np.isfinite(value) or not 0 < value < .08:
        raise ValueError('Selected grasp command width must be finite in (0, .08)')
    return float(value)


def _rectangle_xy(center, axis, along, across):
    axis = np.asarray(axis, dtype=float)[:2]
    axis /= np.linalg.norm(axis)
    normal = np.array([-axis[1], axis[0]])
    center = np.asarray(center, dtype=float)[:2]
    return np.array([center+a*along/2*axis+b*across/2*normal
                     for a,b in ((-1,-1),(1,-1),(1,1),(-1,1))], dtype=np.float32)


def verify_destination_jaw_clearance(destination, grasp_width, templates, name,
                                     slot_counts, settings, execution, tcp_closing_axis=(0.,1.,0.)):
    """Check the complete narrow jaw sweep against measured occupied slots.

    Only modeled occupied brick volumes are checked here. Template fixtures and
    unseen obstacles still require measured scene/collision geometry; a sparse
    occupancy count is not evidence that an unobserved region is empty.
    """
    destination = matrix(destination)
    margin = float(execution.get('gripper_opening_margin_m', .004))
    thickness = float(settings.get('finger_clearance_m', .016))
    pad = float(settings.get('finger_pad_length_m', .020))
    z_allowance = float(settings.get('clearance_depth_margin_m', .002))
    xy_allowance = float(settings.get('destination_neighbor_margin_m', .001))
    if (not all(np.isfinite(v) for v in (margin, thickness, pad, z_allowance, xy_allowance)) or
            margin <= 0 or thickness <= 0 or pad <= 0 or z_allowance < 0 or xy_allowance < 0):
        raise ValueError('Physical destination jaw sweep dimensions must be finite and positive')
    if len(slot_counts) != 12 or any(isinstance(n, bool) or not isinstance(n, (int,np.integer)) or n < 0 for n in slot_counts):
        raise ValueError('Destination clearance requires all12 validated occupied slot counts')
    surface = surface_from_tcp(destination, execution['tcp_to_fingertip_m'],
                               execution['finger_insertion_below_surface_m'])
    closing = _unit(destination[:3,:3] @ _unit(tcp_closing_axis))
    if np.linalg.norm(closing[:2]) < .9:
        raise ValueError('Destination closing axis is not near-horizontal')
    closing_xy = closing[:2] / np.linalg.norm(closing[:2])
    pad_xy = np.array([-closing_xy[1], closing_xy[0]])
    sweep = thickness + margin/2
    jaw_polygons = [_rectangle_xy(surface[:2]+side*closing_xy*(grasp_width/2+sweep/2),
                                  pad_xy, pad, sweep) for side in (-1,1)]
    finger_bottom = surface[2]-execution['finger_insertion_below_surface_m']
    checked = []
    for slot_index, count in enumerate(slot_counts):
        if not count:
            continue
        # Only the highest occupied brick can intersect the descending jaw from
        # above; any lower layer is no higher than that top and is below the tip.
        neighbor_pose = matrix(templates.placement(name,(int(count)-1)*12+slot_index)['placement_pose'])
        neighbor_surface = surface_from_tcp(neighbor_pose, execution['tcp_to_fingertip_m'],
                                            execution['finger_insertion_below_surface_m'])
        if neighbor_surface[2]+z_allowance < finger_bottom:
            continue
        neighbor_closing = _unit(neighbor_pose[:3,:3] @ _unit(tcp_closing_axis))
        long_axis = (neighbor_closing if execution['template_grasp_orientation']=='across_length'
                     else np.cross([0.,0.,1.],neighbor_closing))
        polygon = _rectangle_xy(neighbor_surface, long_axis,
                                float(settings.get('expected_length_m', .0318))+2*xy_allowance,
                                float(settings.get('expected_width_m', .0158))+2*xy_allowance)
        checked.append(slot_index)
        for jaw_index, jaw in enumerate(jaw_polygons):
            area, _ = cv2.intersectConvexConvex(jaw, polygon)
            if area > 1e-10:
                raise ValueError(f'Destination jaw sweep intersects occupied slot {slot_index} (jaw {jaw_index}); widen template spacing')
    return dict(checked_slots=checked, opening_width_m=float(grasp_width+margin),
                swept_thickness_m=sweep)


def _project(frame, points):
    k = frame.intrinsics
    if any(abs(c) > 1e-8 for c in k.get('coeffs', [])):
        import pyrealsense2 as rs
        intr = rs.intrinsics()
        for name in ('width','height','fx','fy','ppx','ppy','coeffs'):
            setattr(intr, name, k[name])
        intr.model = getattr(rs.distortion, k['distortion'].split('.')[-1])
        return np.array([rs.rs2_project_point_to_pixel(intr, p.tolist()) for p in points])
    return np.column_stack((points[:,0]/points[:,2]*k['fx']+k['ppx'],
                            points[:,1]/points[:,2]*k['fy']+k['ppy']))


def _depth_changes(before, after, mask, before_camera, after_camera):
    rows, cols = np.nonzero(mask & np.isfinite(before.depth_m) & (before.depth_m > 0))
    step = max(1, len(rows)//400)
    rows, cols = rows[::step], cols[::step]
    if not len(rows):
        raise ValueError('No measured source footprint depth')
    old = np.array([before.deproject(float(x),float(y),float(before.depth_m[y,x]))
                    for y,x in zip(rows,cols)])
    relative = np.linalg.inv(matrix(after_camera)) @ matrix(before_camera)
    predicted = (relative @ np.column_stack((old,np.ones(len(old)))).T).T[:,:3]
    valid_z = predicted[:,2] > .001
    pixels = np.zeros((len(old),2), dtype=float)
    pixels[valid_z] = _project(after, predicted[valid_z])
    finite = np.isfinite(pixels).all(axis=1)
    pixels[~finite] = -1000
    pixels = np.round(pixels).astype(int)
    h,w = after.depth_m.shape
    inside = valid_z & finite & (pixels[:,0]>=0) & (pixels[:,0]<w) & (pixels[:,1]>=0) & (pixels[:,1]<h)
    sampled = np.zeros(len(old),dtype=float)
    sampled[inside] = after.depth_m[pixels[inside,1],pixels[inside,0]]
    valid = inside & np.isfinite(sampled) & (sampled > 0)
    return sampled[valid]-predicted[valid,2], float(np.count_nonzero(valid)/len(old))


def verify_source_removal(before, after, candidate, before_camera, after_camera, settings):
    """Require exposed deeper source pixels and unchanged surrounding support."""
    if before.depth_m.shape != after.depth_m.shape:
        raise ValueError('Source verification image dimensions changed')
    footprint = np.zeros(before.depth_m.shape,np.uint8)
    cv2.fillConvexPoly(footprint,np.round(candidate.box_px).astype(np.int32),255)
    inner = cv2.erode(footprint,np.ones((5,5),np.uint8)) > 0
    ring = (cv2.dilate(footprint,np.ones((15,15),np.uint8)) > 0) & (footprint == 0)
    changes, valid = _depth_changes(before,after,inner,before_camera,after_camera)
    min_valid = float(settings.get('source_verification_valid_fraction',.75))
    min_removed = float(settings.get('source_removed_fraction',.8))
    minimum = float(settings.get('min_source_depth_increase_m',.004))
    if len(changes) < 25 or valid < min_valid:
        raise RuntimeError('Selected source footprint is occluded or has insufficient verification depth')
    if not 0 < min_removed <= 1 or not np.isfinite(minimum) or minimum <= 0:
        raise ValueError('Invalid source removal thresholds')
    fraction = float(np.mean(changes >= minimum))
    if fraction < min_removed:
        raise RuntimeError(f'Source brick removal unverified: deeper fraction={fraction:.3f}')
    surrounding, ring_valid = _depth_changes(before,after,ring,before_camera,after_camera)
    tolerance = float(settings.get('source_surrounding_depth_tolerance_m',.004))
    stable = float(np.mean(np.abs(surrounding) <= tolerance)) if len(surrounding) else 0.
    if len(surrounding) < 25 or ring_valid < min_valid or stable < float(settings.get('source_surrounding_stable_fraction',.75)):
        raise RuntimeError('Source surroundings changed or are occluded; possible attached-neighbor motion')
    maximum = settings.get('max_source_depth_increase_m')
    if maximum is not None and float(np.median(changes)) > float(maximum):
        raise RuntimeError('Source depth change exceeds measured individual-brick removal allowance')
    return dict(removed_fraction=fraction, median_depth_increase_m=float(np.median(changes)),
                valid_fraction=valid, surrounding_stable_fraction=stable)


class PalletRunner(SortRunner):
    """Reuse tested bounded transit methods; use pallet-specific observations."""
    def __init__(self, config, poses, tcp_color, model, robot, camera, templates):
        if not isinstance(templates, TemplateRegistry):
            raise ValueError('Pallet execution requires a measured TemplateRegistry')
        self.c,self.e,self.robot,self.camera = config,config['execution'],robot,camera
        self.poses = {name:matrix(value).copy() for name,value in poses.items()}
        self.tcp_color,self.model,self.templates = matrix(tcp_color),model,templates
        self.planner = SortPlanner(config['job'],config['journal_file'],templates)
        self.settings = dict(config.get('perception',{}))
        self.settings.update(config.get('pallet',{}))
        self.tcp_above_surface_m = grasp_tcp_surface_offset(self.e)
        # The depth-based jaw-footprint check needs physical insertion, not TCP offset.
        self.settings['grasp_depth_below_surface_m'] = self.e['finger_insertion_below_surface_m']
        self.settings['jaw_opening_margin_m'] = self.e.get('gripper_opening_margin_m', .004)
        self.preflight()

    def preflight(self):
        if self.e.get('source_arrangement') not in ('pallet','pallet_unconnected'):
            raise ValueError('Pallet runner requires source_arrangement=pallet; connected bricks need a verified separator strategy')
        if self.c['job'].get('source_count') is None and self.c['job'].get('requested_count') is None:
            raise ValueError('Bound the pallet job with source_count or requested_count')
        self.low,self.high = (np.asarray(self.e.get(k),float) for k in ('workspace_min_m','workspace_max_m'))
        if (self.low.shape != (3,) or self.high.shape != (3,) or
                not np.isfinite(self.low).all() or not np.isfinite(self.high).all() or np.any(self.low>=self.high)):
            raise ValueError('Teach finite measured workspace bounds')
        self.transit = float(self.e['transit_height_m'])
        if not np.isfinite(self.transit) or not self.low[2]<self.transit<=self.high[2]:
            raise ValueError('Transit height outside measured workspace')
        for name in ('consensus_frames','max_observation_attempts','max_pallet_layers'):
            value = self.e.get(name)
            if isinstance(value,bool) or not isinstance(value,int) or value<1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('settle_sec','tcp_to_fingertip_m','finger_insertion_below_surface_m','detection_position_tolerance_m',
                     'destination_xy_tolerance_m','destination_z_tolerance_m'):
            value = self.e.get(name)
            if value is None or not np.isfinite(value) or value < 0 or (name != 'settle_sec' and value == 0):
                raise ValueError(f'{name} must be finite and positive')
        self.tcp_above_surface_m = grasp_tcp_surface_offset(self.e)
        maximum_insertion = self.e.get('maximum_usable_tip_insertion_m')
        if (isinstance(maximum_insertion, bool) or maximum_insertion is None or
                not np.isfinite(maximum_insertion) or not 0 < maximum_insertion < .0096):
            raise ValueError('maximum_usable_tip_insertion_m must be an explicit finite physical measurement below9.6mm')
        if self.e['finger_insertion_below_surface_m'] > maximum_insertion:
            raise ValueError('Finger insertion exceeds the measured usable tip before the obstructing screw')
        opening_margin = self.e.get('gripper_opening_margin_m', .004)
        if isinstance(opening_margin, bool) or not np.isfinite(opening_margin) or not 0 < opening_margin <= .02:
            raise ValueError('gripper_opening_margin_m must be finite in (0, .02]')
        tracking_tolerance = self.e.get('gripper_opening_tracking_tolerance_m', min(.0001,opening_margin/4))
        if (isinstance(tracking_tolerance,bool) or not np.isfinite(tracking_tolerance) or
                not 0 < tracking_tolerance <= opening_margin/2):
            raise ValueError('gripper_opening_tracking_tolerance_m must be positive and no larger than half the narrow opening margin')
        if self.settings.get('grasp_width_source', 'measured') not in ('measured', 'configured_2x4_nominal'):
            raise ValueError('grasp_width_source must be measured or configured_2x4_nominal')
        gap = self.settings.get('measured_interbrick_gap_m')
        if gap is not None:
            thickness = self.settings.get('finger_clearance_m', .016)
            if (isinstance(gap, bool) or not np.isfinite(gap) or gap <= 0 or
                    isinstance(thickness, bool) or not np.isfinite(thickness) or thickness <= 0):
                raise ValueError('Measured gap and physical finger clearance must be finite and positive')
            if thickness+opening_margin/2 > gap+1e-12:
                raise ValueError('Finger thickness plus per-jaw opening sweep exceeds measured interbrick gap')
        # These gates are measurement requirements, not best-effort scores.
        # Reject malformed thresholds before a transfer can start.
        for name, default in (('source_verification_valid_fraction', .75),
                              ('source_removed_fraction', .8),
                              ('source_surrounding_stable_fraction', .75)):
            value = self.settings.get(name, default)
            if isinstance(value, bool) or not np.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f'{name} must be finite in (0, 1]')
        fraction = self.e.get('destination_surface_fraction', .7)
        if isinstance(fraction, bool) or not np.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError('destination_surface_fraction must be finite in (0, 1]')
        for name, default in (('min_source_depth_increase_m', .004),
                              ('source_surrounding_depth_tolerance_m', .004)):
            value = self.settings.get(name, default)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        maximum = self.settings.get('max_source_depth_increase_m')
        if maximum is not None and (isinstance(maximum, bool) or not np.isfinite(maximum) or
                maximum < self.settings.get('min_source_depth_increase_m', .004)):
            raise ValueError('max_source_depth_increase_m must exceed the removal threshold')
        if self.e.get('template_grasp_orientation') not in ('across_width','across_length'):
            raise ValueError('Set measured template_grasp_orientation explicitly')
        if not self.e.get('verify_destination',True):
            raise ValueError('Pallet execution requires destination verification')
        if self.model is None or self.settings.get('roi') is None:
            raise ValueError('Teach physical shade references and an input-only pallet ROI')
        self.model.validate_labels(self.c['job']['good_colors']+self.c['job']['bad_colors'])
        motion = self.c['motion']
        if (self.templates.base_frame != motion['base_frame'] or
                self.templates.tcp_frame != motion['tcp_frame']):
            raise ValueError('Template frames differ from motion TCP/base frames')
        for name in ('observation','pick_reference','scrap'):
            if name not in self.poses:
                raise ValueError(f'Missing taught pose {name}')
            self.check(self.poses[name])
            above(self.poses[name],self.transit)
        optical_down = -(self.poses['observation'] @ self.tcp_color)[2,2]
        if optical_down < self.e.get('min_downward_optical_cosine',.95):
            raise ValueError('Pallet observation camera must point approximately downward')
        for name in {'requested'} | set(self.planner.leftover_routes.values()):
            for slot in range(12):
                pose = self.templates.placement(name,(self.e['max_pallet_layers']-1)*12+slot)['placement_pose']
                self.check(pose)
                if -matrix(pose)[2,2] < .95:
                    raise ValueError('Measured placement orientation must point stock-Hand fingertips downward')
                above(pose,self.transit)

    def _capture(self):
        frame = self.camera.capture_burst()
        actual = self.robot.current_pose()
        return frame,actual

    def _known_candidates(self, inspection, observation):
        candidates=[]
        known=set(self.c['job']['good_colors']+self.c['job']['bad_colors'])
        for candidate in inspection.candidates:
            if candidate.graspable and candidate.color_id in known:
                center,_=candidate_base(candidate,observation,self.tcp_color)
                candidates.append((float(center[2]),candidate))
        return [item[1] for item in sorted(candidates,key=lambda item:(-item[0],-item[1].clearance_fraction))]

    def observe_source(self):
        self.transit_to(self.poses['observation'])
        time.sleep(self.e['settle_sec'])
        failures=[]
        for _ in range(self.e['max_observation_attempts']):
            try:
                first,observation=self._capture()
                candidates=self._known_candidates(inspect_pallet(first,self.settings,self.model),observation)
                if not candidates:
                    raise ValueError('No known individually accessible brick in source ROI')
                selected=candidates[0]
                center,axis=candidate_base(selected,observation,self.tcp_color)
                last,last_pose=first,observation
                for _ in range(self.e['consensus_frames']-1):
                    last,last_pose=self._capture()
                    matches=[]
                    for candidate in self._known_candidates(inspect_pallet(last,self.settings,self.model),last_pose):
                        point,closing=candidate_base(candidate,last_pose,self.tcp_color)
                        angle=math.degrees(math.acos(float(np.clip(abs(np.dot(axis,closing)),0,1))))
                        if (candidate.color_id==selected.color_id and
                                candidate.grasp_orientation==selected.grasp_orientation and
                                np.linalg.norm(point-center)<=self.e['detection_position_tolerance_m'] and
                                abs(candidate.grasp_width_m-selected.grasp_width_m)<=.003 and
                                angle<=float(self.e.get('detection_axis_tolerance_deg',8.))):
                            matches.append((float(np.linalg.norm(point-center)),candidate))
                    if not matches:
                        raise ValueError('Same accessible source brick not stable across observations')
                    selected=min(matches,key=lambda item:item[0])[1]
                return selected,last,last_pose
            except ValueError as error:
                failures.append(str(error))
        raise RuntimeError('Pallet observation failed: '+'; '.join(failures))

    def _verify_destination(self,view,expected_surface,color,long_axis):
        # A registered slot remains known when adjacent equal-color bricks merge
        # visually. Verify the measured slot footprint instead of inventing a
        # segmentation boundary through a growing single-color ridge.
        self.transit_to(view)
        time.sleep(self.e['settle_sec'])
        failures=[]
        long_axis=_unit(long_axis)
        cross=_unit(np.cross([0.,0.,1.],long_axis))
        length=float(self.settings.get('expected_length_m',.0318))
        width=float(self.settings.get('expected_width_m',.0158))
        vertices=np.array([expected_surface+a*length/2*long_axis+b*width/2*cross
                           for a,b in ((-1,-1),(1,-1),(1,1),(-1,1))])
        for _ in range(self.e['max_observation_attempts']):
            frame,actual=self._capture()
            try:
                self.model.validate_camera(frame.metadata)
                camera=actual @ self.tcp_color
                in_camera=(np.linalg.inv(camera) @ np.column_stack((vertices,np.ones(4))).T).T[:,:3]
                if np.any(in_camera[:,2]<=0):
                    raise ValueError('Registered destination footprint is behind camera')
                pixels=_project(frame,in_camera)
                h,w=frame.depth_m.shape
                if (not np.isfinite(pixels).all() or np.any(pixels[:,0]<0) or np.any(pixels[:,0]>=w)
                        or np.any(pixels[:,1]<0) or np.any(pixels[:,1]>=h)):
                    raise ValueError('Registered destination footprint is clipped')
                mask=np.zeros(frame.depth_m.shape,np.uint8)
                cv2.fillConvexPoly(mask,np.round(pixels).astype(np.int32),255)
                mask=cv2.erode(mask,np.ones((3,3),np.uint8))
                rows,cols=np.nonzero((mask>0)&np.isfinite(frame.depth_m)&(frame.depth_m>0))
                if len(rows)<25 or len(rows)/max(1,np.count_nonzero(mask))<.75:
                    raise ValueError('Insufficient visible measured destination surface')
                measured=np.array([frame.deproject(float(x),float(y),float(frame.depth_m[y,x]))
                                   for y,x in zip(rows,cols)])
                world=(camera @ np.column_stack((measured,np.ones(len(measured)))).T).T[:,:3]
                top=np.abs(world[:,2]-expected_surface[2])<=self.e['destination_z_tolerance_m']
                if np.mean(top)<float(self.e.get('destination_surface_fraction',.7)):
                    raise ValueError('Destination top surface is absent at registered placement height')
                measured_center=np.median(world[top],axis=0)
                if np.linalg.norm(measured_center[:2]-expected_surface[:2])>self.e['destination_xy_tolerance_m']:
                    raise ValueError('Destination measured top center differs from registered slot')
                shade_mask=np.zeros(frame.depth_m.shape,np.uint8)
                shade_mask[rows[top],cols[top]]=255
                result=self.model.classify(frame.color_bgr,shade_mask)
                if result.label!=color:
                    raise ValueError('Destination top shade differs from transferred shade')
                return
            except ValueError as error:
                failures.append(str(error))
        raise RuntimeError('Destination placement verification failed: '+'; '.join(failures))

    def run(self,max_transfers=None):
        completed=0
        while not self.planner.complete and (max_transfers is None or completed<max_transfers):
            candidate,before,observation=self.observe_source()
            center,_=candidate_base(candidate,observation,self.tcp_color)
            target=candidate_pick_pose(candidate,observation,self.tcp_color,self.poses['pick_reference'],
                         self.e['tcp_to_fingertip_m'], self.e['finger_insertion_below_surface_m'],
                         self.c['motion'].get('tcp_closing_axis',[0.,1.,0.]))
            width=selected_grasp_width(candidate,self.settings)
            release_width=width+float(self.e.get('gripper_opening_margin_m', .004))
            if not 0<width<release_width<=min(.08,float(self.c['motion']['open_width_m'])):
                raise ValueError('Measured selected grasp width plus narrow-opening margin exceeds allowed opening')
            self.check(target)
            check_grasp_insertion(target,center,self.e['tcp_to_fingertip_m'],
                self.e['maximum_usable_tip_insertion_m'],release_width,
                self.settings.get('finger_pad_length_m',.020),self.settings.get('finger_clearance_m',.016),
                self.e.get('minimum_grasp_penetration_m',.0002))
            transfer=self.planner.reserve(candidate.color_id)
            slot=transfer['destination']
            try:
                if slot=='scrap':
                    destination=self.poses['scrap'].copy()
                else:
                    if transfer['layer_index']>=self.e['max_pallet_layers']:
                        raise ValueError(f'{slot} measured pallet layer capacity reached')
                    destination=matrix(transfer['placement_pose']).copy()
                    expected_surface=surface_from_tcp(destination, self.e['tcp_to_fingertip_m'],
                                                       self.e['finger_insertion_below_surface_m'])
                    if candidate.grasp_orientation != self.e['template_grasp_orientation']:
                        destination[:3,:3]=_rz(np.pi/2) @ destination[:3,:3]
                        # Rotate around the represented physical contact point, not
                        # around an offset TCP that would displace the brick center.
                        destination=tcp_at_surface(expected_surface, destination, self.e['tcp_to_fingertip_m'],
                                                   self.e['finger_insertion_below_surface_m'])
                self.check(destination)
                if slot!='scrap':
                    verify_destination_jaw_clearance(destination, width, self.templates, slot,
                        self.planner.state['occupancy'][slot]['slot_counts'], self.settings, self.e,
                        self.c['motion'].get('tcp_closing_axis',[0.,1.,0.]))
                above(target,self.transit)
                above(destination,self.transit)
                expected_surface=surface_from_tcp(destination, self.e['tcp_to_fingertip_m'],
                                                  self.e['finger_insertion_below_surface_m'])
                view=self.poses['observation'].copy()
                view[:3,3]+=expected_surface-center
                if slot!='scrap':
                    self.check(view)
                    above(view,self.transit)
            except Exception:
                self.planner.abort(transfer['id'])
                raise
            LOG.info('PALLET TRANSFER id=%s color=%s destination=%s slot=%s layer=%s width=%.4f orientation=%s',
                     transfer['id'],candidate.color_id,slot,transfer['slot_index'],transfer['layer_index'],width,candidate.grasp_orientation)
            LOG.info('GRASP WIDTH source=%s command=%.6fm measured=%.6fm',
                     self.settings.get('grasp_width_source','measured'),width,
                     float(getattr(candidate,'measured_grasp_width_m',None) or candidate.grasp_width_m))
            self.planner.mark_motion_started(transfer['id'])
            self.robot.open_gripper(width_m=release_width)
            tracking_tolerance=self.e.get('gripper_opening_tracking_tolerance_m',
                                         min(.0001,(release_width-width)/4))
            preopen_check=verify_narrow_preopen(float(self.robot.gripper_state()['width_m']),width,
                                               release_width,self.settings,tracking_tolerance)
            LOG.info('MEASURED PREOPEN VERIFIED BEFORE DESCENT %s',preopen_check)
            self.transit_to(above(target,self.transit))
            self.linear(target)
            # The controller's endpoint tolerance can exceed the entire usable
            # shallow toe. Recompute physical corner insertion from fresh actual
            # TCP and opening measurements immediately before closing.
            actual_opening=float(self.robot.gripper_state()['width_m'])
            verify_narrow_preopen(actual_opening,width,release_width,self.settings,tracking_tolerance)
            insertion_check=check_grasp_insertion(self.robot.current_pose(),center,
                self.e['tcp_to_fingertip_m'],self.e['maximum_usable_tip_insertion_m'],actual_opening,
                self.settings.get('finger_pad_length_m',.020),self.settings.get('finger_clearance_m',.016),
                self.e.get('minimum_grasp_penetration_m',.0002))
            LOG.info('PRECLOSE FINGERTIPS VERIFIED %s',insertion_check)
            if not self.robot.grasp(width_m=width):
                raise RuntimeError('Pallet grasp failed; inspect pending transfer')
            self.linear(above(target,self.transit))
            if abs(self.robot.gripper_state()['width_m']-width)>self.c['motion']['gripper_width_tolerance_m']:
                raise RuntimeError('Measured held width differs from selected brick grasp')
            self.transit_to(self.poses['observation'])
            time.sleep(self.e['settle_sec'])
            after,after_pose=self._capture()
            metrics=verify_source_removal(before,after,candidate,observation @ self.tcp_color,
                                          after_pose @ self.tcp_color,self.settings)
            LOG.info('SOURCE VERIFIED %s',metrics)
            self.transit_to(above(destination,self.transit))
            self.linear(destination)
            self.robot.open_gripper(width_m=release_width)
            self.linear(above(destination,self.transit))
            if slot!='scrap':
                closing=destination[:3,:3] @ _unit(self.c['motion'].get('tcp_closing_axis',[0.,1.,0.]))
                long_axis=(closing if candidate.grasp_orientation=='across_length'
                           else np.cross([0.,0.,1.],closing))
                self._verify_destination(view,expected_surface,candidate.color_id,long_axis)
            self.planner.commit(transfer['id'],verified_release=True)
            completed+=1
            LOG.info('PALLET COMMITTED occupancy=%s composition=%s',self.planner.state['occupancy'],self.planner.composition)
        return self.planner.state
