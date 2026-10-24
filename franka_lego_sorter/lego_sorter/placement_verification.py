"""Read-only commissioning evidence for one observed placement; never commits counts."""
import cv2
import numpy as np


def _transform(points,T):
    points=np.asarray(points,float)
    return points@np.asarray(T,float)[:3,:3].T+np.asarray(T,float)[:3,3]


def _project(points,intrinsics):
    points=np.asarray(points,float)
    if np.any(points[:,2]<=0):raise ValueError('Footprint behind camera')
    return np.column_stack((points[:,0]/points[:,2]*intrinsics['fx']+intrinsics['ppx'],points[:,1]/points[:,2]*intrinsics['fy']+intrinsics['ppy']))


def verify_placement(before,after,T_base_color_before,T_base_color_after,slot,model,held_color='6117418'):
    """Compare independently observed depth over a registered physical footprint.

    `slot` is one measured observed_slots item, not an inferred lattice target.
    Candidate extrinsics remain commissioning-only. Return evidence, never a
    production occupancy/count update or a calibration verification flag.
    """
    model.validate_camera(before.metadata);model.validate_camera(after.metadata)
    point=np.asarray(slot['point_camera_m'],float);normal=np.asarray(slot['surface_normal_camera'],float);normal/=np.linalg.norm(normal)
    corners=[]
    for pixel in slot['box_px']:
        ray=before.deproject(*map(float,pixel),1.);corners.append(ray*float(normal@point)/float(normal@ray))
    world_corners=_transform(corners,T_base_color_before);world_point=_transform([point],T_base_color_before)[0]
    world_normal=np.asarray(T_base_color_before)[:3,:3]@normal;world_normal/=np.linalg.norm(world_normal)
    after_inverse=np.linalg.inv(T_base_color_after)
    def sample(frame,polygon,T):
        h,w=frame.depth_m.shape
        if np.any(polygon[:,0]<0) or np.any(polygon[:,0]>=w) or np.any(polygon[:,1]<0) or np.any(polygon[:,1]>=h):raise ValueError('Observed footprint outside image; reobserve')
        mask=np.zeros((h,w),np.uint8);cv2.fillConvexPoly(mask,np.round(polygon).astype(int),255);mask=cv2.erode(mask,np.ones((7,7),np.uint8));valid=(mask>0)&np.isfinite(frame.depth_m)&(frame.depth_m>0)
        fraction=float(valid.sum()/max(1,(mask>0).sum()))
        if fraction<.8 or valid.sum()<40:raise ValueError('Insufficient actual depth over placement footprint')
        ys,xs=np.nonzero(valid);z=frame.depth_m[ys,xs];cloud=np.column_stack(((xs-frame.intrinsics['ppx'])*z/frame.intrinsics['fx'],(ys-frame.intrinsics['ppy'])*z/frame.intrinsics['fy'],z));world=_transform(cloud,T);heights=(world-world_point)@world_normal
        center=float(np.median(heights));mad=float(np.median(abs(heights-center)));shade=model.classify(frame.color_bgr,mask)
        return dict(median_height_m=center,depth_mad_m=mad,depth_valid_fraction=fraction,point_count=int(len(heights)),color_id=shade.label,color_distance=shade.distance,color_margin=shade.margin,feature_lab=shade.feature_lab),mask
    before_sample,_=sample(before,np.asarray(slot['box_px'],float),T_base_color_before)
    polygon=_project(_transform(world_corners,after_inverse),after.intrinsics)
    # Reproject observed footprint at its actually measured upper height, rather
    # than pretending camera pixels at the old support height are the new face.
    for _ in range(2):
        after_sample,mask=sample(after,polygon,T_base_color_after)
        polygon=_project(_transform(world_corners+world_normal*after_sample['median_height_m'],after_inverse),after.intrinsics)
    after_sample,mask=sample(after,polygon,T_base_color_after)
    increment=after_sample['median_height_m']-before_sample['median_height_m'];reasons=[]
    if after_sample['color_id']!=str(held_color):reasons.append('released brick color unknown or differs from operator-confirmed held color')
    if increment<=.002:reasons.append('no resolved positive height increment above2mm diagnostic floor')
    if max(before_sample['depth_mad_m'],after_sample['depth_mad_m'])>.0015:reasons.append('observed surface depth spread too broad for useful increment')
    return dict(format='readonly_single_placement_evidence_v1',status='consistent_with_held_brick_placement' if not reasons else 'UNKNOWN',reasons=reasons,held_color_id=str(held_color),before=before_sample,after=after_sample,measured_visible_top_increment_m=increment,after_observed_footprint_px=polygon.tolist(),world_reference_normal=world_normal.tolist(),execution_eligible=False,production_counts_committed=False,measured_layer_pitch_verified=False,limitations='One observed upper-surface increment under candidate extrinsics; body/stud semantics and absolute calibration uncertainty unresolved. Cannot establish repeated-layer pitch, exact brick count, stud engagement or long-term retention.'),mask
