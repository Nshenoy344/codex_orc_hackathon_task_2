"""Measure a destination template: four observed ridges of three observed slots.

No spacing, missing slot, fixture pocket, or physical template type is invented.
Brick templates use independently observed brick faces. Base/fixture templates
need twelve operator-demarcated slot polygons in the actual aligned RGB-D image.
All positions below are measured support surfaces, not robot TCP targets.
"""
from dataclasses import dataclass, asdict
import cv2
import numpy as np
from .pallet import inspect_pallet, fit_surface_plane


class TemplateError(ValueError):
    pass


@dataclass
class TemplateSlot:
    slot_id: str
    ridge: int
    index_in_ridge: int
    point_camera_m: list
    long_axis_camera: list
    surface_normal_camera: list
    length_m: float
    width_m: float
    center_px: list
    box_px: list
    surface_plane_rms_m: float


@dataclass
class MeasuredTemplate:
    template_type: str
    slots: list
    long_axis_camera: list
    ridge_axis_camera: list
    surface_normal_camera: list
    ridge_gaps_m: list
    within_ridge_gaps_m: list
    support_height_above_table_m: float | None
    placement_ready: bool
    metrics: dict

    def to_dict(self):
        return dict(template_type=self.template_type, slots=[asdict(s) for s in self.slots],
                    long_axis_camera=self.long_axis_camera, ridge_axis_camera=self.ridge_axis_camera,
                    surface_normal_camera=self.surface_normal_camera, ridge_gaps_m=self.ridge_gaps_m,
                    within_ridge_gaps_m=self.within_ridge_gaps_m,
                    support_height_above_table_m=self.support_height_above_table_m,
                    placement_ready=self.placement_ready, metrics=self.metrics)


def _unit(vector):
    vector = np.asarray(vector, float)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm < 1e-8:
        raise TemplateError('Invalid template orientation vector')
    return vector/norm


def _stable_sign(axis):
    index = int(np.argmax(np.abs(axis[:2])))
    return axis if axis[index] >= 0 else -axis


def _as_observation(value):
    return asdict(value) if hasattr(value, '__dataclass_fields__') else dict(value)


def group_measured_slots(observations, config=None):
    """Validate/group 12 measured faces; retain every center and measured gap."""
    config = config or {}
    kind = config.get('template_type')
    if kind not in ('brick_layer', 'fixture', 'base'):
        raise TemplateError('Specify physical template_type: brick_layer, fixture, or base')
    items = [_as_observation(value) for value in observations]
    if len(items) != 12:
        raise TemplateError(f'Exactly 12 independently observed slots required; found {len(items)}')
    positions = np.array([item['point_camera_m'] for item in items], float)
    axes = np.array([_unit(item['long_axis_camera']) for item in items])
    normals = np.array([_unit(item['surface_normal_camera']) for item in items])
    if positions.shape != (12, 3) or not np.isfinite(positions).all():
        raise TemplateError('Template centers must be finite measured 3D points')
    length = np.array([item['length_m'] for item in items], float)
    width = np.array([item['width_m'] for item in items], float)
    if not np.isfinite(length).all() or not np.isfinite(width).all() or np.any(length <= 0) or np.any(width <= 0):
        raise TemplateError('Template slot dimensions must be finite measured positive lengths')
    plane_rms = np.array([item.get('surface_plane_rms_m', np.nan) for item in items], float)
    if not np.isfinite(plane_rms).all() or np.any(plane_rms > float(config.get('max_slot_plane_rms_m', .003))):
        raise TemplateError('Each slot needs a reliable measured support plane')
    tolerance = float(config.get('size_tolerance_m', .004))
    if np.any(np.abs(length-float(config.get('expected_length_m', .0318))) > tolerance) or np.any(np.abs(width-float(config.get('expected_width_m', .0158))) > tolerance):
        raise TemplateError('Observed slot dimensions do not match measured 2x4 footprint')
    normal = _unit(normals.mean(axis=0))
    _, vecs = np.linalg.eigh(axes.T@axes)
    along = _stable_sign(_unit(vecs[:, -1]-normal*np.dot(vecs[:, -1], normal)))
    cross = _stable_sign(_unit(np.cross(normal, along)))
    angle = np.degrees(np.arccos(np.clip(np.abs(axes@along), 0, 1)))
    if angle.max() > float(config.get('max_axis_deviation_deg', 8.)):
        raise TemplateError('The twelve observed bricks/slots do not form parallel ridges')
    normal_angle = np.degrees(np.arccos(np.clip(normals@normal, -1, 1)))
    if normal_angle.max() > float(config.get('max_normal_deviation_deg', 8.)):
        raise TemplateError('Template support faces have inconsistent normals')
    signed_heights = (positions-positions.mean(axis=0))@normal
    if np.ptp(signed_heights) > float(config.get('max_support_height_span_m', .004)):
        raise TemplateError('Template slots are not on one measured support layer')
    a, b = positions@along, positions@cross
    # Grouping partitions only independently detected centers. No centers are
    # generated, snapped onto a grid, averaged into place, or forced equal-spaced.
    cross_order = np.argsort(b)
    groups = [cross_order[index:index+3] for index in range(0, 12, 3)]
    group_centers = []
    for group in groups:
        if np.ptp(b[group]) > float(config.get('max_ridge_transverse_span_m', .004)):
            raise TemplateError('Observed centers cannot be grouped into four ridges of three')
        group_centers.append(float(np.mean(b[group])))
    ridge_gaps = []
    for previous, following in zip(groups[:-1], groups[1:]):
        gap = float(np.mean(b[following])-np.mean(b[previous])-
                    (np.mean(width[following])+np.mean(width[previous]))/2)
        if gap < -float(config.get('max_overlap_m', .002)):
            raise TemplateError('Adjacent ridge footprints overlap or ridge assignment is ambiguous')
        ridge_gaps.append(gap)
    slots, longitudinal_gaps = [], []
    for ridge, group in enumerate(groups):
        order = group[np.argsort(a[group])]
        gaps = []
        for prior, following in zip(order[:-1], order[1:]):
            gap = float(a[following]-a[prior]-(length[following]+length[prior])/2)
            if gap < -float(config.get('max_overlap_m', .002)):
                raise TemplateError('Slots overlap along a ridge; geometry/boundaries are ambiguous')
            gaps.append(gap)
        longitudinal_gaps.append(gaps)
        for index, source_index in enumerate(order):
            item = items[source_index]
            axis = _unit(item['long_axis_camera'])
            if np.dot(axis, along) < 0:
                axis = -axis
            slots.append(TemplateSlot(f'r{ridge}b{index}', ridge, index,
                         positions[source_index].tolist(), axis.tolist(),
                         _unit(item['surface_normal_camera']).tolist(),
                         float(length[source_index]), float(width[source_index]),
                         list(item.get('center_px', [])), list(item.get('box_px', [])),
                         float(item.get('surface_plane_rms_m', 0.))))
    verified = config.get('template_type_verified') is True
    return MeasuredTemplate(kind, slots, along.tolist(), cross.tolist(), normal.tolist(), ridge_gaps,
                            longitudinal_gaps, None, False,
                            dict(observed_slots=12, ridges=4, slots_per_ridge=3,
                                 template_type_verified=verified, table_height_measured=False,
                                 max_axis_deviation_deg=float(angle.max()),
                                 support_height_span_m=float(np.ptp(signed_heights)),
                                 ridge_center_coordinates_m=group_centers,
                                 support_semantics=('Existing brick-layer top surfaces; next output layer is above these measured faces'
                                                    if kind == 'brick_layer' else
                                                    'Measured base/pocket support surfaces; fixture top is not automatically the brick support'),
                                 incomplete_reasons=['table/support height not measured']+([] if verified else ['physical template type unverified'])))


def _mask(shape, polygon):
    vertices = np.asarray(polygon, float)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
        raise TemplateError('Station/slot polygon must contain >=3 measured pixel vertices')
    h, w = shape
    if not np.isfinite(vertices).all() or np.any(vertices[:, 0] < 0) or np.any(vertices[:, 1] < 0) or np.any(vertices[:, 0] >= w) or np.any(vertices[:, 1] >= h):
        raise TemplateError('Template polygon is clipped/outside image')
    result = np.zeros(shape, np.uint8)
    cv2.fillPoly(result, [np.round(vertices).astype(np.int32)], 255)
    if np.count_nonzero(result) < 25:
        raise TemplateError('Template polygon too small')
    return result


def _measure_slot_polygon(frame, polygon, config):
    vertices = np.asarray(polygon, np.float32)
    if len(vertices) != 4 or not cv2.isContourConvex(vertices.reshape(-1, 1, 2)):
        raise TemplateError('Fixture/base slot needs four visible convex footprint corners')
    region = _mask(frame.depth_m.shape, polygon)
    interior = cv2.erode(region, np.ones((3, 3), np.uint8)) > 0
    valid = (interior & np.isfinite(frame.depth_m) & (frame.depth_m > float(config.get('min_depth_m', .15))) &
             (frame.depth_m < float(config.get('max_depth_m', 1.5))))
    if np.count_nonzero(valid)/max(1, np.count_nonzero(interior)) < float(config.get('min_slot_depth_fraction', .75)):
        raise TemplateError('Insufficient measured depth inside a template slot')
    center3d, normal, rms = fit_surface_plane(frame, valid)
    if rms > float(config.get('max_slot_plane_rms_m', .003)):
        raise TemplateError('Fixture/base slot surface is not a reliable measured plane')
    def on_plane(pixel):
        ray = frame.deproject(float(pixel[0]), float(pixel[1]), 1.)
        denominator = float(normal@ray)
        if abs(denominator) < .1:
            raise TemplateError('Template slot plane is too oblique')
        return ray*float(normal@center3d)/denominator
    corners = np.array([on_plane(pixel) for pixel in vertices])
    # Operator corners are the actual footprint, not a bounding box spanning gaps.
    edges = [((corners[(i+1)%4]-corners[i])+(corners[(i+2)%4]-corners[(i+3)%4]))/2 for i in range(2)]
    lengths = [float(np.linalg.norm(edge)) for edge in edges]
    along_index = int(np.argmax(lengths))
    pixel_center = vertices.mean(axis=0)
    return dict(point_camera_m=on_plane(pixel_center).tolist(),
                long_axis_camera=_unit(edges[along_index]).tolist(), surface_normal_camera=normal.tolist(),
                length_m=max(lengths), width_m=min(lengths), center_px=pixel_center.tolist(),
                box_px=vertices.tolist(), surface_plane_rms_m=rms)


def register_template(frame, config, station_polygon_px=None, candidates=None):
    """Measure template geometry from a real aligned Frame and explicit semantics.

    For fixture/base, config.slot_polygons_px must hold 12 independently observed
    footprint polygons; no lattice extrapolation is performed. For brick_layer,
    exactly 12 complete brick faces must be observed within the station polygon.
    config.table_polygon_px supplies surrounding table/support-reference depth.
    """
    kind = config.get('template_type')
    if kind not in ('brick_layer', 'fixture', 'base'):
        raise TemplateError('Explicit template_type required: brick_layer, fixture, base')
    polygon = station_polygon_px if station_polygon_px is not None else config.get('station_polygon_px')
    if polygon is None:
        raise TemplateError('An operator-demarcated destination station polygon is required')
    station_mask = _mask(frame.depth_m.shape, polygon)
    if kind == 'brick_layer':
        if candidates is None:
            settings = dict(config)
            ys, xs = np.nonzero(station_mask)
            settings['roi'] = [int(xs.min()), int(ys.min()), int(np.ptp(xs)+1), int(np.ptp(ys)+1)]
            candidates = inspect_pallet(frame, settings, model=None).candidates
        items = []
        rejected = []
        allowed = {'physical_shade_unverified', 'neighbor_or_unknown_depth_in_jaw_strip', 'jaw_footprint_outside_image'}
        for candidate in candidates:
            item = _as_observation(candidate)
            center = np.round(item['center_px']).astype(int)
            if not (0 <= center[0] < frame.depth_m.shape[1] and 0 <= center[1] < frame.depth_m.shape[0]) or station_mask[center[1], center[0]] == 0:
                continue
            box = np.asarray(item['box_px'], float)
            for vertex in box:
                if cv2.pointPolygonTest(np.asarray(polygon, np.float32), tuple(map(float, vertex)), False) < 0:
                    raise TemplateError('A template brick crosses the station boundary; inspect full footprint')
            invalid = set(item.get('rejection_reasons', []))-allowed
            if item.get('split_region') or invalid:
                rejected.append(sorted(invalid or {'same-color split boundary unverified'}))
                continue
            items.append(item)
        if rejected:
            raise TemplateError(f'Ambiguous/incomplete brick faces inside template station: {rejected}')
    else:
        polygons = config.get('slot_polygons_px')
        if polygons is None or len(polygons) != 12:
            raise TemplateError('Fixture/base requires exactly 12 individually observed slot_polygons_px')
        items = []
        for slot_polygon in polygons:
            for vertex in slot_polygon:
                if cv2.pointPolygonTest(np.asarray(polygon, np.float32), tuple(map(float, vertex)), False) < 0:
                    raise TemplateError('A measured slot falls outside destination station polygon')
            items.append(_measure_slot_polygon(frame, slot_polygon, config))
    result = group_measured_slots(items, config)
    table_polygon = config.get('table_polygon_px')
    if table_polygon is not None:
        region = cv2.erode(_mask(frame.depth_m.shape, table_polygon), np.ones((3, 3), np.uint8)) > 0
        valid = region & np.isfinite(frame.depth_m) & (frame.depth_m > float(config.get('min_depth_m', .15))) & (frame.depth_m < float(config.get('max_depth_m', 1.5)))
        if np.count_nonzero(valid)/max(1, np.count_nonzero(region)) < .8:
            raise TemplateError('Surrounding table polygon has insufficient valid depth')
        point, normal, rms = fit_surface_plane(frame, valid)
        if rms > float(config.get('max_table_plane_rms_m', .003)):
            raise TemplateError('Surrounding table is not a reliable measured plane')
        heights = np.array([(np.array(slot.point_camera_m)-point)@normal for slot in result.slots])
        if heights.min() < -float(config.get('max_below_table_m', .003)):
            raise TemplateError('Template support lies below measured table; check station/table polygons')
        result.support_height_above_table_m = float(np.median(heights))
        result.metrics.update(table_height_measured=True, table_plane_rms_m=rms,
                              table_plane_point_camera_m=point.tolist(), table_plane_normal_camera=normal.tolist(),
                              support_height_range_m=[float(heights.min()), float(heights.max())])
        result.metrics['incomplete_reasons'] = ([] if result.metrics['template_type_verified'] else ['physical template type unverified'])
    result.placement_ready = result.metrics['template_type_verified'] and result.metrics['table_height_measured']
    result.metrics.update(camera_metadata=frame.metadata, camera_intrinsics=frame.intrinsics,
                          captured_monotonic=frame.captured_monotonic)
    result.metrics['vertical_layer_increment_measured'] = False
    result.metrics['note'] = 'No vertical layer pitch inferred. Measure an actual subsequent placed layer before continuing upward.'
    return result
