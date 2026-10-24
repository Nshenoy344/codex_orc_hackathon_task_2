"""Conditional static released-brick envelope and exact stock-Hand primitive review."""
from functools import lru_cache
from itertools import product
from pathlib import Path
import struct
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import matrix
from .environment_boxes import intersections


def static_bounds(release,target):
    anchor=matrix(release['T_base_tcp_before']);bottom=np.array(target['estimated_bottom_footprint_tcp_m']);normal=np.array(target['bottom_plane_tcp']['normal'])
    vertices=np.vstack((bottom,bottom+normal*.0114));settling=np.array([0.,0.,-.003])@anchor[:3,:3]
    union=np.vstack((vertices,vertices+settling));return anchor,union.min(0)-.001,union.max(0)+.001


@lru_cache(maxsize=1)
def palm_vertices():
    raw=Path('/home/nshenoy/franka_ros2_ws/src/franka_description/meshes/robot_ee/franka_hand_white/collision/hand.stl').read_bytes();n=struct.unpack_from('<I',raw,80)[0];dtype=np.dtype([('normal','<f4',3),('vertices','<f4',(3,3)),('attribute','<u2')]);return np.frombuffer(raw,dtype=dtype,count=n,offset=84)['vertices'].reshape(-1,3).astype(float)-[0,0,.1034]


def hand_parts(opening):
    parts=[('palm',palm_vertices())];boxes=[([0,.0185,.011],[.022,.015,.020],0.),([0,.0068,.0022],[.022,.0088,.0038],0.),([0,.0159,.02835],[.0175,.007,.0235],np.pi/6),([0,.00758,.04525],[.0175,.0152,.0185],0.)]
    for side in(-1,1):
        for i,(center,size,rx)in enumerate(boxes):
            corners=np.array(list(product(*[(-v/2,v/2)for v in size])))@Rotation.from_rotvec([rx,0,0]).as_matrix().T+center
            if side<0:corners=corners@np.diag([-1.,-1.,1.])
            corners+=np.array([0.,side*opening/2,.0584-.1034]);parts.append((f'finger{side}_part{i}',corners))
    return parts


def review_hand(poses,opening,anchor,lo,hi,camera_extrinsic=None):
    parts=hand_parts(opening);hits=[]
    if camera_extrinsic is not None:
        from .contact_camera_envelope import COLOR_HOUSING_LOWER_M,COLOR_HOUSING_UPPER_M
        x=matrix(camera_extrinsic);camera=np.array(list(product(*zip(COLOR_HOUSING_LOWER_M-.002,COLOR_HOUSING_UPPER_M+.002))))
        parts.append(('camera_assumed_housing',camera@x[:3,:3].T+x[:3,3]))
        a=x[:3,3];b=np.array([0.,0.,-.1034]);parts.append(('bracket_conservative_bounds',np.array(list(product(*zip(np.minimum(a,b)-.027,np.maximum(a,b)+.027))))))
    for i,pose in enumerate(poses):
        for name,vertices in parts:
            world=vertices@pose[:3,:3].T+pose[:3,3];relative=(world-anchor[:3,3])@anchor[:3,:3]
            for _ in intersections(relative,[('released_brick',lo,hi)]):hits.append({'sample_index':i,'part':name})
    return {'static_brick_hand_intersections':hits,'static_brick_hand_clear':not hits,'nominal_centered_inner_jaw_gap_per_side_m':(float(opening)-.0158)/2,
        'static_box_lower_release_tcp_m':lo.tolist(),'static_box_upper_release_tcp_m':hi.tolist(),'static_box_uncertainty_m':.001,'settling_down_range_m':[0.,.003],
        'semantics':'Fixed conditional released-object envelope; no attached object. Exact stockHand collision primitives receive no additional padding here because released-envelope already includes1mm geometry margin; ordinary cloud review retains2mm margin.'}
