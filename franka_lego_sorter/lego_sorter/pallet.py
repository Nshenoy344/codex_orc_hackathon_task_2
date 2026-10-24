"""Offline/live multi-brick pallet inspection with conservative jaw clearance.

Segmentation clusters generate geometry only. Physical shade IDs require a taught
ColorModel. A candidate is not executable unless geometry and both jaw strips
have adequate measured clearance. This module never commands a robot.
"""
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import cv2
import numpy as np
from .camera import Frame
from .perception import lab_feature, roi_mask


@dataclass
class PalletCandidate:
    center_px: list
    box_px: list
    point_camera_m: list
    long_axis_camera: list
    length_m: float
    width_m: float
    depth_m: float
    feature_lab: list
    color_id: str | None
    color_distance: float | None
    color_margin: float | None
    graspable: bool
    rejection_reasons: list
    clearance_fraction: float
    split_region: bool
    grasp_width_m: float
    grasp_closing_axis_camera: list
    grasp_orientation: str
    clearance_options: dict
    surface_normal_camera: list
    surface_plane_rms_m: float
    grasp_width_source: str = 'measured'
    measured_grasp_width_m: float | None = None
    surface_reference: str = 'unverified_face'
    body_top_diagnostics: dict | None = None


@dataclass
class PalletInspection:
    candidates: list
    diagnostics: dict

    def to_dict(self):
        return dict(candidates=[asdict(candidate) for candidate in self.candidates],
                    diagnostics=self.diagnostics)


def _polygon_mask(shape, vertices):
    mask = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(mask, np.round(vertices).astype(np.int32), 255)
    return mask


def jaw_clearance(frame, center, long_axis_px, length_px, width_px, depth_m, config,
                  surface_normal_camera=None, surface_point_camera=None):
    """Check two horizontal jaw footprints at the commanded grasp depth.

    Measured points below the observed local face plane by the required finger
    insertion are free. Raw opticalZ is not a physical height on an oblique view.
    Missing depth and image boundaries are unknown and fail closed. This checks visible top-down
    clearance, not unseen side geometry or interlocking studs.
    """
    axis = np.asarray(long_axis_px, float)
    axis /= max(np.linalg.norm(axis), 1e-9)
    normal = np.array([-axis[1], axis[0]])
    scale = (frame.intrinsics['fx'] + frame.intrinsics['fy']) / (2*depth_m)
    opening_margin = float(config.get('jaw_opening_margin_m', .004))
    if not np.isfinite(opening_margin) or opening_margin < 0:
        raise ValueError('jaw_opening_margin_m must be finite and nonnegative')
    # The outermost swept finger edge occurs after each jaw has opened by half
    # the total width margin. Checking only closed-finger thickness misses it.
    finger = (float(config.get('finger_clearance_m', .010))+opening_margin/2)*scale
    pad = float(config.get('finger_pad_length_m', .020))*scale
    grasp_depth = float(config.get('grasp_depth_below_surface_m', .006))
    uncertainty = float(config.get('clearance_depth_margin_m', .002))
    normal3 = np.asarray([0.,0.,-1.] if surface_normal_camera is None else surface_normal_camera,dtype=float)
    point3 = np.asarray(frame.deproject(float(center[0]),float(center[1]),depth_m)
                        if surface_point_camera is None else surface_point_camera,dtype=float)
    if normal3.shape != (3,) or point3.shape != (3,) or not np.isfinite(normal3).all() or not np.isfinite(point3).all() or np.linalg.norm(normal3)<1e-9:
        raise ValueError('Jaw clearance requires a valid observed surface plane')
    normal3/=np.linalg.norm(normal3)
    if normal3[2]>0:normal3=-normal3
    values = []
    reasons = []
    for side in (-1, 1):
        # Inner edge begins immediately outside the brick boundary; outer edge
        # includes physical finger thickness plus approach allowance and half
        # the commanded total opening margin (the complete opening/closing sweep).
        a = np.asarray(center)+normal*side*(width_px/2+1)
        b = a+normal*side*finger
        corners = np.array([a-axis*pad/2, a+axis*pad/2, b+axis*pad/2, b-axis*pad/2])
        h, w = frame.depth_m.shape
        if np.any(corners[:, 0] < 0) or np.any(corners[:, 1] < 0) or np.any(corners[:, 0] >= w) or np.any(corners[:, 1] >= h):
            reasons.append('jaw_footprint_outside_image')
            values.append(0.)
            continue
        mask = _polygon_mask(frame.depth_m.shape, corners) > 0
        rows,cols=np.nonzero(mask)
        depth=frame.depth_m[rows,cols]
        valid=np.isfinite(depth)&(depth>0)
        free=np.zeros(len(depth),dtype=bool)
        if np.any(valid):
            z=depth[valid]
            if any(abs(c)>1e-8 for c in frame.intrinsics.get('coeffs',[])):
                measured=np.array([frame.deproject(float(x),float(y),float(value))
                    for x,y,value in zip(cols[valid],rows[valid],z)])
            else:
                k=frame.intrinsics
                measured=np.column_stack(((cols[valid]-k['ppx'])/k['fx']*z,
                                           (rows[valid]-k['ppy'])/k['fy']*z,z))
            signed_height=(measured-point3)@normal3
            free[valid]=signed_height < -(grasp_depth+uncertainty)
        fraction = float(np.count_nonzero(free)/max(1, len(depth)))
        values.append(fraction)
        if fraction < float(config.get('min_jaw_clearance_fraction', .90)):
            reasons.append('neighbor_or_unknown_depth_in_jaw_strip')
    return min(values), sorted(set(reasons))


def fit_surface_plane(frame, mask):
    """Robust local 3D plane; normal faces toward the observing camera."""
    rows, cols = np.nonzero(mask)
    step = max(1, len(rows)//400)
    points = np.array([frame.deproject(float(x), float(y), float(frame.depth_m[y, x]))
                       for y, x in zip(rows[::step], cols[::step])])
    if len(points) < 10:
        raise ValueError('Too few depth points for a local surface plane')
    chosen = points
    for _ in range(3):
        center = np.median(chosen, axis=0)
        _, _, vt = np.linalg.svd(chosen-center, full_matrices=False)
        normal = vt[-1]
        errors = np.abs((points-center)@normal)
        cutoff = max(.0015, float(np.median(errors))*2.5)
        subset = points[errors <= cutoff]
        if len(subset) < 10:
            break
        chosen = subset
    if normal[2] > 0:
        normal = -normal
    rms = float(np.sqrt(np.mean(((chosen-center)@normal)**2)))
    return center, normal, rms


def fit_brick_body_top(frame, center_px, long_axis_px, length_px, width_px, approximate_depth_m, config=None):
    """Measure body-top plane between stud rows; never subtract nominal stud height.

    A2x4's nominal stud pattern is used only to select independent sampling
    regions. The strip plane must be measured from depth, and raised stud-center
    samples must actually resolve a plausible positive relief. Unresolved relief
    is an unverified reference, not a reason to invent a corrected surface.
    """
    config = config or {}
    center = np.asarray(center_px,dtype=float)
    axis = np.asarray(long_axis_px,dtype=float)
    axis /= max(np.linalg.norm(axis),1e-12)
    cross = np.array([-axis[1],axis[0]])
    scale = (frame.intrinsics['fx']+frame.intrinsics['fy'])/(2*approximate_depth_m)
    half_strip = float(config.get('body_top_strip_half_width_m',.00065))*scale
    if not 0 < half_strip < width_px/4:
        raise ValueError('Body-top strip is not inside the observed brick face')
    vertices = np.array([center+a*length_px*.40*axis+b*half_strip*cross
                         for a,b in ((-1,-1),(1,-1),(1,1),(-1,1))])
    strip = _polygon_mask(frame.depth_m.shape,vertices)>0
    depth = frame.depth_m
    valid = np.isfinite(depth)&(depth>0)&(np.abs(depth-approximate_depth_m)<=float(config.get('body_top_search_band_m',.006)))
    support = strip&valid
    fraction = np.count_nonzero(support)/max(1,np.count_nonzero(strip))
    strip_area = float(config.get('body_top_min_strip_area_m2',.000010))
    stud_area = float(config.get('body_top_min_stud_area_m2',.000001))
    if not np.isfinite(strip_area) or not np.isfinite(stud_area) or strip_area<=0 or stud_area<=0:
        raise ValueError('Body/stud minimum measured support areas must be positive and finite')
    minimum_strip_pixels=max(int(config.get('body_top_min_strip_pixels',40)),int(np.ceil(strip_area*scale**2)))
    minimum_stud_pixels=max(int(config.get('body_top_min_stud_pixels',5)),int(np.ceil(stud_area*scale**2)))
    if np.count_nonzero(support)<minimum_strip_pixels or fraction<float(config.get('body_top_min_strip_valid_fraction',.85)):
        raise ValueError('Insufficient reliable depth in the measured inter-stud body strip')
    # The narrow strip determines ONLY the body-plane offset. Its width is
    # insufficient to identify a reliable second slope under quantized stereo
    # depth: an unconstrained strip SVD can manufacture opposite stud-row heights.
    # Estimate orientation independently over the observed wide face instead.
    wide_vertices=np.array([center+a*length_px*.44*axis+b*width_px*.42*cross
                            for a,b in ((-1,-1),(1,-1),(1,1),(-1,1))])
    wide_mask=(_polygon_mask(depth.shape,wide_vertices)>0)&valid
    wide_rows,wide_cols=np.nonzero(wide_mask)
    if len(wide_rows)<max(40,minimum_strip_pixels):
        raise ValueError('Too little independently measured wide-face support for the body-plane normal')
    _,normal,wide_rms=fit_surface_plane(frame,wide_mask)
    wide_step=max(1,len(wide_rows)//400)
    wide_points=np.array([frame.deproject(float(x),float(y),float(depth[y,x]))
                          for y,x in zip(wide_rows[::wide_step],wide_cols[::wide_step])])
    _,wide_singular,_=np.linalg.svd(wide_points-wide_points.mean(axis=0),full_matrices=False)
    wide_spread=wide_singular/np.sqrt(len(wide_points))
    if wide_spread[1]<float(config.get('body_top_min_normal_support_std_m',.002)):
        raise ValueError('Wide face does not independently constrain both body-plane slopes')
    if wide_rms>float(config.get('body_top_max_normal_support_rms_m',.0015)):
        raise ValueError('Wide face is not a reliable independent body-plane normal support')
    rows,cols=np.nonzero(support)
    strip_points=np.array([frame.deproject(float(x),float(y),float(depth[y,x])) for y,x in zip(rows,cols)])
    projected_height=float(np.median(strip_points@normal))
    body_center=np.median(strip_points,axis=0)
    body_center+=normal*(projected_height-float(body_center@normal))
    rms=float(np.sqrt(np.mean((strip_points@normal-projected_height)**2)))
    if rms>float(config.get('body_top_max_plane_rms_m',.001)):
        raise ValueError('Measured inter-stud strip is not a sufficiently stable body-plane offset')
    relief=[]
    pixel_counts=[]
    stud_centers=[]
    for along in (-.012,-.004,.004,.012):
        for across in (-.004,.004):
            point=center+scale*(along*axis+across*cross)
            region=np.zeros(depth.shape,np.uint8)
            cv2.circle(region,tuple(np.round(point).astype(int)),max(1,round(.0011*scale)),255,-1)
            mask=(region>0)&valid
            rows,cols=np.nonzero(mask)
            pixel_counts.append(len(rows));stud_centers.append(point.tolist())
            if len(rows)<minimum_stud_pixels:
                relief.append(None)
                continue
            measured=np.array([frame.deproject(float(x),float(y),float(depth[y,x])) for y,x in zip(rows,cols)])
            relief.append(float(np.median((measured-body_center)@normal)))
    low=float(config.get('body_top_min_resolved_stud_relief_m',.0008))
    high=float(config.get('body_top_max_resolved_stud_relief_m',.0028))
    if not 0<low<high:
        raise ValueError('Measured stud-relief acceptance band must be positive and ordered')
    accepted=[value for value in relief if value is not None and low<=value<=high]
    available=[value for value in relief if value is not None]
    diagnostics=dict(reference='measured_inter_stud_body_top',strip_pixel_count=int(np.count_nonzero(support)),
        strip_valid_fraction=float(fraction),body_plane_rms_m=float(rms),strip_polygon_px=vertices.tolist(),
        minimum_strip_pixels=minimum_strip_pixels,minimum_stud_pixels=minimum_stud_pixels,
        plane_normal_source='independently_measured_wide_face',body_strip_fits_offset_only=True,
        wide_normal_support_pixels=len(wide_rows),wide_normal_sample_count=len(wide_points),wide_normal_support_std_m=wide_spread.tolist(),
        wide_normal_support_rms_m=float(wide_rms),body_plane_normal_camera=normal.tolist(),
        support_minimum_physical_areas_m2=dict(strip=strip_area,stud=stud_area),
        measured_stud_relief_m=relief,stud_sample_pixel_counts=pixel_counts,stud_centers_px=stud_centers,
        required_relief_band_m=[low,high],resolved_stud_count=len(accepted),
        nominal_pattern_used_only_for_sampling=True,nominal_height_subtracted=False,
        uncertainty='RMS/relief distinguish surfaces; absolute depth and hand-eye accuracy remain independently unverified')
    required=int(config.get('body_top_min_resolved_studs',6))
    if not 1<=required<=8:
        raise ValueError('body_top_min_resolved_studs must lie in[1,8]')
    verified=len(accepted)>=required and len(available)>=required
    diagnostics['verified']=bool(verified)
    if not verified:
        diagnostics['reason']='Measured depth does not resolve body-top versus raised stud surfaces'
    return body_center,normal,rms,diagnostics


def inspect_pallet(frame, config=None, model=None):
    config = config or {}
    width_source = config.get('grasp_width_source', 'measured')
    if width_source not in ('measured', 'configured_2x4_nominal'):
        raise ValueError('grasp_width_source must be measured or configured_2x4_nominal')
    color, depth = frame.color_bgr, frame.depth_m
    if color.shape[:2] != depth.shape:
        raise ValueError('Pallet inspection requires aligned color/depth')
    if model is not None:
        model.validate_camera(frame.metadata)
    h, w = depth.shape
    roi = config.get('roi')
    source = roi_mask(color.shape, roi) > 0 if roi else np.ones(depth.shape, bool)
    valid = (source & np.isfinite(depth) & (depth > float(config.get('min_depth_m', .15))) &
             (depth < float(config.get('max_depth_m', 1.0))))
    if np.count_nonzero(valid) < 100:
        raise ValueError('No valid pallet depth')
    table = float(config.get('table_depth_m') or np.percentile(depth[valid], 85))
    object_mask = valid & (depth < table-float(config.get('min_object_height_m', .008)))
    if np.count_nonzero(object_mask) < 100:
        raise ValueError('No raised pallet surface; specify table_depth_m or include table in ROI')
    smooth = cv2.bilateralFilter(color, 9, 25, 9)
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)
    # Exclude black holes/deep seams. Stud shadows stay within the color cluster
    # after closing, while true region boundaries retain chromatic separation.
    chroma = np.linalg.norm(lab[:, :, 1:].astype(float)-128., axis=2)
    eligible = (object_mask & (lab[:, :, 0] > int(config.get('min_lightness', 30))) &
                (chroma >= float(config.get('min_chroma', 16.))))
    pixels = lab[eligible].astype(np.float32)
    if len(pixels) < 100:
        raise ValueError('Too few usable color/depth pixels')
    k = min(int(config.get('segmentation_clusters', 7)), max(2, len(pixels)//100))
    cv2.setRNGSeed(17)
    _, labels, centers = cv2.kmeans(pixels, k, None,
                                  (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 50, .5),
                                  3, cv2.KMEANS_PP_CENTERS)
    segmentation = np.full(depth.shape, -1, np.int16)
    segmentation[eligible] = labels.ravel()
    expected_long = float(config.get('expected_length_m', .0318))
    expected_short = float(config.get('expected_width_m', .0158))
    tolerance = float(config.get('size_tolerance_m', .006))
    candidates = []
    rejected_regions = 0
    for cluster in range(k):
        mask = (segmentation == cluster).astype(np.uint8)*255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask[~object_mask] = 0
        count, cc, stats, _ = cv2.connectedComponentsWithStats(mask)
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] < int(config.get('min_component_px', 150)):
                continue
            component = (cc == index).astype(np.uint8)*255
            contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = max(contours, key=cv2.contourArea)
            (cx, cy), (rw, rh), angle = cv2.minAreaRect(contour)
            box = cv2.boxPoints(((cx, cy), (rw, rh), angle))
            edge0, edge1 = box[1]-box[0], box[2]-box[1]
            edge = edge0 if np.linalg.norm(edge0) >= np.linalg.norm(edge1) else edge1
            lp, sp = max(rw, rh), min(rw, rh)
            if sp < 4 or lp < 8:
                continue
            normal = np.array([-edge[1], edge[0]])/max(np.linalg.norm(edge), 1e-9)
            axis = edge/max(np.linalg.norm(edge), 1e-9)
            support = (component > 0) & valid
            z = float(np.median(depth[support]))
            center_point = frame.deproject(cx, cy, z)
            long_vector = frame.deproject(cx+axis[0]*lp/2, cy+axis[1]*lp/2, z)-frame.deproject(cx-axis[0]*lp/2, cy-axis[1]*lp/2, z)
            short_vector = frame.deproject(cx+normal[0]*sp/2, cy+normal[1]*sp/2, z)-frame.deproject(cx-normal[0]*sp/2, cy-normal[1]*sp/2, z)
            lm, sm = np.linalg.norm(long_vector), np.linalg.norm(short_vector)
            # Same-color joined blocks may form a 2x8+ region. They are proposed
            # as tiles for inspection but explicitly remain geometrically ambiguous.
            pieces = max(1, int(round(lm/expected_long)))
            if pieces > int(config.get('max_region_tiles', 6)) or abs(sm-expected_short) > tolerance or abs(lm/pieces-expected_long) > tolerance:
                rejected_regions += 1
                continue
            tile_lp = lp/pieces
            for tile in range(pieces):
                center = np.array([cx, cy])+axis*(tile+.5-pieces/2)*tile_lp
                vertices = np.array([center-axis*tile_lp/2-normal*sp/2,
                                     center+axis*tile_lp/2-normal*sp/2,
                                     center+axis*tile_lp/2+normal*sp/2,
                                     center-axis*tile_lp/2+normal*sp/2])
                interior = cv2.erode(_polygon_mask(depth.shape, vertices), np.ones((5, 5), np.uint8))
                pixel_mask = (interior > 0) & support
                if np.count_nonzero(pixel_mask) < 25:
                    continue
                tile_z = float(np.median(depth[pixel_mask]))
                plane_center, plane_normal, plane_rms = fit_surface_plane(frame, pixel_mask)
                center_ray = frame.deproject(*center, 1.)
                plane_denominator = float(plane_normal @ center_ray)
                if abs(plane_denominator) > .1:
                    tile_z = float((plane_normal @ plane_center)/plane_denominator)
                surface_reference='unverified_face'
                body_diagnostics={}
                body_verified=False
                try:
                    body_center,body_normal,body_rms,body_diagnostics=fit_brick_body_top(
                        frame,center,axis,tile_lp,sp,tile_z,config)
                    body_verified=body_diagnostics['verified']
                    if body_verified:
                        denominator=float(body_normal@center_ray)
                        if abs(denominator)<.1:
                            raise ValueError('Measured body plane is too oblique for an accurate center')
                        tile_z=float((body_normal@body_center)/denominator)
                        plane_normal,plane_rms=body_normal,body_rms
                        surface_reference='measured_inter_stud_body_top'
                except ValueError as error:
                    body_diagnostics=dict(verified=False,reason=str(error),nominal_height_subtracted=False)
                reasons = []
                if config.get('require_body_top_reference',True) and not body_verified:
                    reasons.append('body_top_plane_unverified')
                if plane_rms > float(config.get('max_face_plane_rms_m', .004)):
                    reasons.append('face_depth_not_planar')
                if pieces > 1:
                    reasons.append('same_color_region_split_boundary_unverified')
                border = float(config.get('image_border_px', 4))
                if (np.any(vertices[:, 0] < border) or np.any(vertices[:, 1] < border) or
                        np.any(vertices[:, 0] > w-1-border) or np.any(vertices[:, 1] > h-1-border)):
                    reasons.append('brick_clipped_by_image')
                support_fraction = float(np.count_nonzero(pixel_mask)/max(1, np.count_nonzero(interior)))
                if support_fraction < float(config.get('min_face_support_fraction', .65)):
                    reasons.append('nonrectangular_or_occluded_face')
                feature = lab_feature(color, pixel_mask.astype(np.uint8)*255)
                label = distance = margin = None
                if model is not None:
                    result = model.classify(color, pixel_mask.astype(np.uint8)*255)
                    label, distance, margin = result.label, result.distance, result.margin
                if label is None:
                    reasons.append('physical_shade_unverified')
                # The measured contour must pass the size gate above, but a
                # narrow physical opening cannot be based on its noisy width.
                # In nominal mode position both jaw strips from the verified
                # nominal2x4 body edges, retaining all measured dimensions.
                width_grasp = float(expected_short if width_source == 'configured_2x4_nominal' else sm)
                length_grasp = float(expected_long if width_source == 'configured_2x4_nominal' else lm/pieces)
                width_px = sp*width_grasp/float(sm)
                length_px = tile_lp*length_grasp/float(lm/pieces)
                plane_point=frame.deproject(*center,tile_z)
                across_width = jaw_clearance(frame, center, axis, tile_lp, width_px, tile_z, config,
                                             plane_normal,plane_point)
                across_length = jaw_clearance(frame, center, normal, sp, length_px, tile_z, config,
                                              plane_normal,plane_point)
                options = {'across_width': dict(free_fraction=across_width[0], reasons=across_width[1],
                                                body_width_m=width_grasp, measured_width_m=float(sm),
                                                grasp_width_source=width_source,clearance_reference='signed_distance_to_observed_face_plane'),
                           'across_length': dict(free_fraction=across_length[0], reasons=across_length[1],
                                                 body_width_m=length_grasp, measured_width_m=float(lm/pieces),
                                                 grasp_width_source=width_source,clearance_reference='signed_distance_to_observed_face_plane')}
                orientation = 'across_width' if across_width[0] >= across_length[0] else 'across_length'
                clearance, clearance_reasons = across_width if orientation == 'across_width' else across_length
                measured_grasp_width = float(sm if orientation == 'across_width' else lm/pieces)
                grasp_width = width_grasp if orientation == 'across_width' else length_grasp
                closing_pixel_axis = normal if orientation == 'across_width' else axis
                closing_vector = frame.deproject(*(center+closing_pixel_axis), tile_z)-frame.deproject(*(center-closing_pixel_axis), tile_z)
                closing_vector /= max(np.linalg.norm(closing_vector), 1e-9)
                if grasp_width > float(config.get('max_gripper_width_m', .08)):
                    reasons.append('brick_exceeds_gripper_width')
                reasons.extend(clearance_reasons)
                p = frame.deproject(*center, tile_z)
                world_axis = frame.deproject(*(center+axis*tile_lp/2), tile_z)-frame.deproject(*(center-axis*tile_lp/2), tile_z)
                world_axis /= max(np.linalg.norm(world_axis), 1e-9)
                candidates.append(PalletCandidate(center.tolist(), vertices.tolist(), p.tolist(), world_axis.tolist(),
                                  float(lm/pieces), float(sm), tile_z, feature.tolist(), label, distance,
                                  margin, not reasons, sorted(set(reasons)), clearance, pieces > 1,
                                  grasp_width, closing_vector.tolist(), orientation, options,
                                  plane_normal.tolist(), plane_rms, width_source, measured_grasp_width,
                                  surface_reference, body_diagnostics))
    candidates.sort(key=lambda candidate: (candidate.depth_m, candidate.center_px[1], candidate.center_px[0]))
    return PalletInspection(candidates, dict(table_depth_m=table, object_depth_range_m=[float(np.min(depth[object_mask])),
                            float(np.max(depth[object_mask]))], segmentation_clusters=k,
                            candidate_count=len(candidates), graspable_count=sum(c.graspable for c in candidates),
                            rejected_regions=rejected_regions,
                            body_top_verified_count=sum(c.surface_reference=='measured_inter_stud_body_top' for c in candidates),
                            body_top_reference_required=bool(config.get('require_body_top_reference',True)),
                            assumption='Near-normal view; unknown shades and unverified split boundaries cannot execute',
                            metadata=frame.metadata))


def annotate(frame, inspection):
    image = frame.color_bgr.copy()
    for index, candidate in enumerate(inspection.candidates):
        color = (0, 220, 0) if candidate.graspable else (0, 160, 255)
        cv2.polylines(image, [np.round(candidate.box_px).astype(np.int32)], True, color, 2)
        label = f'{index}: {candidate.color_id or "?"} {candidate.depth_m:.3f}m {candidate.grasp_orientation[-5:]}'
        cv2.putText(image, label, tuple(np.round(candidate.center_px).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1)
    return image


def main():
    # Offline analysis should not launch a broad OpenCV worker pool beside1kHz control.
    cv2.setNumThreads(1)
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot')
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--output', default='data/pallet_inspection')
    args = parser.parse_args()
    import yaml
    config = {}
    if args.config:
        config = yaml.safe_load(Path(args.config).read_text())
    from .perception import ColorModel
    model = ColorModel.load(args.model) if args.model else None
    data = np.load(args.snapshot, allow_pickle=False)
    frame = Frame(data['color_bgr'], data['depth_m'], json.loads(str(data['intrinsics_json'])),
                  float(data['captured_monotonic']), json.loads(str(data['metadata_json'])))
    settings = dict(config.get('perception', config))
    settings.update(config.get('pallet', {}))
    execution = config.get('execution', {})
    if 'finger_insertion_below_surface_m' in execution:
        settings['grasp_depth_below_surface_m'] = execution['finger_insertion_below_surface_m']
    if 'gripper_opening_margin_m' in execution:
        settings['jaw_opening_margin_m'] = execution['gripper_opening_margin_m']
    inspection = inspect_pallet(frame, settings, model)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix('.json').write_text(json.dumps(inspection.to_dict(), indent=2))
    cv2.imwrite(str(output.with_suffix('.png')), annotate(frame, inspection))
    print(json.dumps(inspection.diagnostics, indent=2))


if __name__ == '__main__':
    main()
