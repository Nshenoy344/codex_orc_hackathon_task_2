"""Offline conservative wrist-camera envelope for a future isolated contact review.

Bounds are explicit assumptions, NOT vendor CAD or measured mount geometry.
This module sends no actions and does not authorize gripper closure or a lift.
"""
from itertools import product
import numpy as np
from .geometry import matrix

COLOR_HOUSING_LOWER_M=np.array([-.100,-.025,-.040])
COLOR_HOUSING_UPPER_M=np.array([ .100, .025, .020])
VENDOR_MODEL_URL="https://github.com/realsenseai/realsense-ros/blob/ros2-master/realsense2_description/urdf/_d435.urdf.xacro"
VENDOR_DEPTH_OPTICAL_CENTER_M=np.array([.0175,0.,0.])
VENDOR_DEPTH_OPTICAL_DIMENSIONS_M=np.array([.090,.025,.02505])
BRACKET_RADIUS_M=.025


def review_contact_camera_envelope(poses,T_tcp_color,cloud_base,table_normal,table_point,margin_m=.002):
    poses=[matrix(p)for p in poses];extrinsic=matrix(T_tcp_color);cloud=np.asarray(cloud_base,float)
    normal=np.asarray(table_normal,float);point=np.asarray(table_point,float)
    if not poses or cloud.ndim!=2 or cloud.shape[1]!=3 or not len(cloud)or not np.isfinite(cloud).all():raise ValueError('Need poses and a nonempty finite observed scene cloud')
    if normal.shape!=(3,)or point.shape!=(3,)or not np.isfinite(normal).all()or not np.isfinite(point).all()or np.linalg.norm(normal)<.99:raise ValueError('Need measured finite table normal/point')
    normal=normal/np.linalg.norm(normal)
    if normal[2]<0:normal=-normal
    if not np.isfinite(margin_m)or not 0<=margin_m<=.005:raise ValueError('Envelope margin must be0..5mm')
    lo=COLOR_HOUSING_LOWER_M-margin_m;hi=COLOR_HOUSING_UPPER_M+margin_m
    corners=np.array(list(product(*zip(lo,hi))));samples=[];radius=BRACKET_RADIUS_M+margin_m
    for i,tcp in enumerate(poses):
        camera=tcp@extrinsic;local=(cloud-camera[:3,3])@camera[:3,:3]
        intersections=int(np.count_nonzero(np.all((local>=lo)&(local<=hi),axis=1)))
        world=corners@camera[:3,:3].T+camera[:3,3]
        camera_table=float(((world-point)@normal).min())
        a=camera[:3,3];b=tcp[:3,3]+tcp[:3,:3]@np.array([0.,0.,-.1034]);delta=b-a
        selected=cloud[np.all((cloud>=np.minimum(a,b)-radius)&(cloud<=np.maximum(a,b)+radius),axis=1)]
        u=np.clip((selected-a)@delta/(delta@delta),0.,1.)
        bracket_intersections=int(np.count_nonzero(np.linalg.norm(selected-(a+u[:,None]*delta),axis=1)<=radius))
        bracket_table=float(min((a-point)@normal,(b-point)@normal)-radius)
        samples.append(dict(sample_index=i,camera_box_observed_intersections=intersections,
            camera_box_table_clearance_m=camera_table,bracket_capsule_observed_intersections=bracket_intersections,
            bracket_capsule_table_clearance_m=bracket_table))
    return dict(vendor_model_url=VENDOR_MODEL_URL,vendor_depth_optical_collision_center_m=VENDOR_DEPTH_OPTICAL_CENTER_M.tolist(),vendor_depth_optical_collision_dimensions_m=VENDOR_DEPTH_OPTICAL_DIMENSIONS_M.tolist(),assumptions='Conservative assumed COLOR OPTICAL frame housing bounds enclosing vendor collision-box Z extent, not measured mount CAD; includes off-center RGB optical origin. Bracket capsule connects optical origin to stock hand origin. Cables and unseen scene are not certified.',
        housing_lower_color_m=COLOR_HOUSING_LOWER_M.tolist(),housing_upper_color_m=COLOR_HOUSING_UPPER_M.tolist(),
        bracket_radius_m=BRACKET_RADIUS_M,additional_margin_m=float(margin_m),samples=samples,
        maximum_camera_observed_intersections=max(s['camera_box_observed_intersections']for s in samples),
        minimum_camera_table_clearance_m=min(s['camera_box_table_clearance_m']for s in samples),
        maximum_bracket_observed_intersections=max(s['bracket_capsule_observed_intersections']for s in samples),
        minimum_bracket_table_clearance_m=min(s['bracket_capsule_table_clearance_m']for s in samples),
        passed=all(s['camera_box_observed_intersections']==0 and s['bracket_capsule_observed_intersections']==0 and s['camera_box_table_clearance_m']>=.005 and s['bracket_capsule_table_clearance_m']>=.005 for s in samples))


def vendor_corners_in_color(T_color_depth):
    transform=matrix(T_color_depth)
    offsets=np.array(list(product(*zip(-VENDOR_DEPTH_OPTICAL_DIMENSIONS_M/2,VENDOR_DEPTH_OPTICAL_DIMENSIONS_M/2))))
    depth=offsets+VENDOR_DEPTH_OPTICAL_CENTER_M
    return depth@transform[:3,:3].T+transform[:3,3]


def verify_vendor_containment(T_color_depth):
    corners=vendor_corners_in_color(T_color_depth)
    contained=bool(np.all(corners>=COLOR_HOUSING_LOWER_M)and np.all(corners<=COLOR_HOUSING_UPPER_M))
    return dict(vendor_model_url=VENDOR_MODEL_URL,vendor_collision_corners_color_m=corners.tolist(),vendor_collision_corner_min_color_m=corners.min(axis=0).tolist(),vendor_collision_corner_max_color_m=corners.max(axis=0).tolist(),assumed_housing_encloses_vendor_box=contained)
