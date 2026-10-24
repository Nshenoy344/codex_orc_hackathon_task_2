#!/usr/bin/env python3
"""Offline dense FK and observed stock-Hand sweep review; never executes a goal."""
import argparse,hashlib,json,struct
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.interpolate import BPoly
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull,cKDTree
import yaml
from lego_sorter.isolated_brick import load_frame,_cloud,_hand_hull,_full_hand_obstructions
from lego_sorter.geometry import matrix


def review(plan_path,urdf_path,snapshot,handeye,geometry, joint_space=False, opening=.080, camera_mode="sphere", include_dense_poses=False, environment_boxes=None):
    if camera_mode not in ("sphere","oriented_contact"):raise ValueError("Unsupported camera envelope mode")
    if not .0<opening<=.0801:raise ValueError('Hand opening outside physical range')
    path=Path(plan_path);data=json.loads(path.read_text());model=ET.parse(urdf_path).getroot()
    joints={x.get('name'):x for x in model.findall('joint')};bychild={x.find('child').get('link'):x for x in model.findall('joint')}
    chain=[];child=data['tcp_frame']
    while child!=data['base_frame']:
        j=bychild[child];chain.append(j);child=j.find('parent').get('link')
    chain.reverse();names=data['joint_names'];points=data['points']
    times=np.array([p['time_sec']for p in points]);spline=BPoly.from_derivatives(times,[[p['positions'],p['velocities'],p['accelerations']]for p in points])
    t=np.unique(np.r_[times,np.linspace(times[0],times[-1],801)]);qs=spline(t);vs=spline.derivative()(t)
    def fk(q,all_links=False):
        values=dict(zip(names,q));result=np.eye(4);link_poses={data['base_frame']:result.copy()}
        for j in chain:
            local=np.eye(4);o=j.find('origin')
            if o is not None:
                local[:3,3]=np.fromstring(o.get('xyz','0 0 0'),sep=' ');local[:3,:3]=Rotation.from_euler('xyz',np.fromstring(o.get('rpy','0 0 0'),sep=' ')).as_matrix()
            if j.get('type')in('revolute','continuous'):
                turn=np.eye(4);turn[:3,:3]=Rotation.from_rotvec(values[j.get('name')]*np.fromstring(j.find('axis').get('xyz'),sep=' ')).as_matrix();local=local@turn
            result=result@local;link_poses[j.find('child').get('link')]=result.copy()
        return link_poses if all_links else result
    poses=np.array([fk(q)for q in qs]);start=matrix(data['T_base_tcp_start']);target=matrix(data['T_base_tcp_target'])
    limits={}
    for k,name in enumerate(names):
        limit=joints[name].find('limit');lo,hi,speed=map(float,(limit.get('lower'),limit.get('upper'),limit.get('velocity')))
        mn,mx,peak=float(qs[:,k].min()),float(qs[:,k].max()),float(np.abs(vs[:,k]).max())
        limits[name]=dict(min_rad=mn,max_rad=mx,lower=lo,upper=hi,peak_speed_rad_s=peak,velocity_limit=speed,passed=mn>=lo-1e-9 and mx<=hi+1e-9 and peak<=speed)
    segment=target[:3,3]-start[:3,3];length=float(np.linalg.norm(segment));direction=segment/length
    projection=(poses[:,:3,3]-start[:3,3])@direction
    line=start[:3,3]+projection[:,None]*direction
    drift=np.linalg.norm(poses[:,:3,3]-line,axis=1)
    angles=np.array([Rotation.from_matrix(start[:3,:3].T@p[:3,:3]).magnitude()for p in poses])
    actual,observation_pose=load_frame(snapshot);X=matrix(yaml.safe_load(Path(handeye).read_text())['T_tcp_color']);B=observation_pose@X
    report_geometry=json.loads(Path(geometry).read_text());brick=np.array(report_geometry['visible_top_base_m'])
    camcloud,_=_cloud(actual,stride=3);cloud=camcloud@B[:3,:3].T+B[:3,3]
    hull=_hand_hull('/home/nshenoy/franka_ros2_ws/src/franka_description/meshes/robot_ee/franka_hand_white/collision/hand.stl')
    steps=np.arange(len(poses)) if joint_space else np.unique(np.linspace(0,len(poses)-1,max(81,int(np.ceil(length/.002))+1)).astype(int));sweep=[]
    for i in steps:
        pose=poses[i]
        # Broad-phase includes complete stock housing/fingers, not merely tips.
        local=(cloud-pose[:3,3])@pose[:3,:3]
        close=np.all((local>[-.04,-.12,-.14])&(local<[.04,.12,.014]),axis=1)
        checked=_full_hand_obstructions(cloud[close],pose,hull,opening=opening,margin=.002)
        sweep.append(dict(sample_index=int(i),time_sec=float(t[i]),observed_points_in_broad_phase=int(close.sum()),**checked))
    arm_sweep=[];environment_hits=[]
    if environment_boxes is not None:
        from lego_sorter.environment_boxes import intersections
    if joint_space:
        # Each visual-environment collision STL is conservatively convexified.
        meshes={}
        for k in range(1,8):
            raw=Path(f'/home/nshenoy/franka_ros2_ws/src/franka_description/meshes/robot_arms/fr3/collision/link{k}.stl').read_bytes()
            n=struct.unpack_from('<I',raw,80)[0]
            dt=np.dtype([('normal','<f4',3),('vertices','<f4',(3,3)),('attribute','<u2')])
            vertices=np.frombuffer(raw,dtype=dt,count=n,offset=84)['vertices'].reshape(-1,3).astype(float)
            hull_arm=ConvexHull(vertices);meshes[f'fr3_link{k}']=(vertices[hull_arm.vertices],hull_arm.equations)
        normal=np.array(report_geometry['measured_table_plane']['normal_base']);table_point=np.array(report_geometry['table_point_base_m'])
        if normal[2]<0:normal=-normal
        arm_steps=np.arange(len(poses))
        for i in arm_steps:
            transforms=fk(qs[i],True)
            for name,(vertices,equations) in meshes.items():
                transform=transforms[name];local=(cloud-transform[:3,3])@transform[:3,:3]
                low=vertices.min(axis=0)-.002;high=vertices.max(axis=0)+.002
                selected=local[np.all((local>=low)&(local<=high),axis=1)]
                count=int(np.all(selected@equations[:,:3].T+equations[:,3]<=.002,axis=1).sum())
                base_vertices=vertices@transform[:3,:3].T+transform[:3,3]
                height=float(((base_vertices-table_point)@normal).min())
                if environment_boxes is not None:
                    for obstacle in intersections(base_vertices,environment_boxes):environment_hits.append(dict(sample_index=int(i),link=name,obstacle=obstacle))
                arm_sweep.append(dict(sample_index=int(i),link=name,observed_points_in_convex_collision_hull=count,minimum_table_plane_clearance_m=height))
    normal=np.array(report_geometry['measured_table_plane']['normal_base']);table_point=np.array(report_geometry['table_point_base_m'])
    if normal[2]<0:normal=-normal
    camera_centers=np.array([(pose@X)[:3,3]for pose in poses])
    hand_origins=poses[:,:3,3]+poses[:,:3,:3]@np.array([0.,0.,-.1034])
    tree=cKDTree(cloud);camera_dist=tree.query(camera_centers)[0]-.060
    camera_table=(camera_centers-table_point)@normal-.060
    bracket_table=np.minimum((camera_centers-table_point)@normal,(hand_origins-table_point)@normal)-.025
    bracket_counts=[]
    for a,b in zip(camera_centers,hand_origins):
        selected=cloud[np.all((cloud>=np.minimum(a,b)-.025)&(cloud<=np.maximum(a,b)+.025),axis=1)]
        delta=b-a;u=np.clip((selected-a)@delta/(delta@delta),0.,1.)
        distance=np.linalg.norm(selected-(a+u[:,None]*delta),axis=1)
        bracket_counts.append(int(np.count_nonzero(distance<=.025)))
    envelopes=dict(assumed_camera_sphere_radius_m=.060,assumed_bracket_capsule_radius_m=.025,
        assumptions='Declared conservative envelopes, not measured CAD; camera sphere at optical origin, bracket capsule to stock hand origin.',
        minimum_camera_observed_cloud_clearance_m=float(camera_dist.min()),minimum_camera_table_clearance_m=float(camera_table.min()),
        maximum_bracket_observed_cloud_intersections=max(bracket_counts),minimum_bracket_table_clearance_m=float(bracket_table.min()))
    envelope_clear=bool(camera_dist.min()>0 and camera_table.min()>=.005 and max(bracket_counts)==0 and bracket_table.min()>=.005)
    if environment_boxes is not None:
        from itertools import product
        hand_corners=np.array(list(product((-.04,.04),(-.12,.12),(-.14,.014))))
        for i,pose in enumerate(poses):
            vertices=hand_corners@pose[:3,:3].T+pose[:3,3]
            for obstacle in intersections(vertices,environment_boxes):environment_hits.append(dict(sample_index=i,link='conservative_full_hand_envelope',obstacle=obstacle))
            from lego_sorter.contact_camera_envelope import COLOR_HOUSING_LOWER_M,COLOR_HOUSING_UPPER_M
            camera=pose@X;camera_corners=np.array(list(product(*zip(COLOR_HOUSING_LOWER_M-.002,COLOR_HOUSING_UPPER_M+.002))))
            camera_vertices=camera_corners@camera[:3,:3].T+camera[:3,3]
            for obstacle in intersections(camera_vertices,environment_boxes):environment_hits.append(dict(sample_index=i,link='conservative_camera_box',obstacle=obstacle))
            a=camera[:3,3];b=pose[:3,3]+pose[:3,:3]@np.array([0.,0.,-.1034])
            # An AABB encloses the declared27mm bracket capsule conservatively.
            bracket_vertices=np.array(list(product(*zip(np.minimum(a,b)-.027,np.maximum(a,b)+.027))))
            for obstacle in intersections(bracket_vertices,environment_boxes):environment_hits.append(dict(sample_index=i,link='conservative_bracket_bounds',obstacle=obstacle))
    camera_helper_hash=None
    if camera_mode=='oriented_contact':
        from lego_sorter.contact_camera_envelope import review_contact_camera_envelope
        envelopes=review_contact_camera_envelope(poses,X,cloud,normal,table_point)
        envelope_clear=bool(envelopes['passed'])
        camera_helper_hash=hashlib.sha256((Path(__file__).resolve().parents[1]/'lego_sorter/contact_camera_envelope.py').read_bytes()).hexdigest()
    tip=poses[:,:3,3]+poses[:,:3,:3]@np.array([0.,0.,.0095])
    clear=all(v['palm_collision_model_points']==0 and v['finger_collision_model_points']==0 for v in sweep)
    report=dict(plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),plan=str(path.resolve()),
        snapshot=str(Path(snapshot).resolve()),snapshot_sha256=hashlib.sha256(Path(snapshot).read_bytes()).hexdigest(),
        handeye=str(Path(handeye).resolve()),handeye_sha256=hashlib.sha256(Path(handeye).read_bytes()).hexdigest(),
        geometry_report=str(Path(geometry).resolve()),geometry_sha256=hashlib.sha256(Path(geometry).read_bytes()).hexdigest(),urdf_sha256=hashlib.sha256(Path(urdf_path).read_bytes()).hexdigest(),reviewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),joint_limits=limits,dense_sample_count=len(t),duration_sec=float(times[-1]),
        start_fk_error_m=float(np.linalg.norm(poses[0,:3,3]-start[:3,3])),endpoint_fk_error_m=float(np.linalg.norm(poses[-1,:3,3]-target[:3,3])),
        endpoint_rotation_error_rad=float(Rotation.from_matrix(poses[-1,:3,:3].T@target[:3,:3]).magnitude()),
        max_cartesian_line_deviation_m=float(drift.max()),path_progress_min_m=float(projection.min()),path_progress_max_m=float(projection.max()),path_length_m=length,
        minimum_tcp_z_m=float(poses[:,2,3].min()),minimum_modeled_tip_height_above_measured_brick_m=float((tip[:,2]-brick[2]).min()),
        maximum_orientation_step_deg=float(np.degrees(angles.max())),full_stock_hand_observed_cloud_sweep=sweep,
        minimum_modeled_tip_table_clearance_m=float(((tip-table_point)@normal).min()),camera_mode=camera_mode,camera_helper_sha256=camera_helper_hash,environment_box_collisions=environment_hits,observed_hand_sweep_clear=clear,hand_opening_assumed_m=opening,camera_bracket_assumed_envelope=envelopes,camera_bracket_envelope_clear=envelope_clear,
        full_arm_convex_hull_review=arm_sweep,full_arm_observed_cloud_clear=all(x['observed_points_in_convex_collision_hull']==0 for x in arm_sweep),full_arm_table_clear=all(x['minimum_table_plane_clearance_m']>=.005 for x in arm_sweep),
        limits_of_review='Numerical candidate handeye and visible cloud only; unseen scene and physical grasp remain unverified; camera/bracket envelopes are conservative declared assumptions, not measured CAD. This is a200mmhover approach only, not descent or grasp authorization.')
    application_branch=all((name not in names or (qs[:,names.index(name)].max()<=hi+1e-9 and (lo is None or qs[:,names.index(name)].min()>=lo-1e-9)))for name,(lo,hi)in {'fr3_joint5':(-1.5,1.5),'fr3_joint6':(None,3.0)}.items())
    checks=dict(recorded_environment_boxes_clear=not environment_hits,application_wrist_branch=bool(application_branch),joint_position_and_speed_limits=all(v['passed']for v in limits.values()),
        complete_planning_fraction=bool(data['fraction']>=.999999),minimum12s_duration=bool(times[-1]>=12.),
        start_fk_consistent=report['start_fk_error_m']<=.0005,endpoint_fk_consistent=report['endpoint_fk_error_m']<=.0005,
        endpoint_rotation_consistent=report['endpoint_rotation_error_rad']<=.002,
        cartesian_line=bool(joint_space or report['max_cartesian_line_deviation_m']<=.0005),
        bounded_progress=bool(joint_space or (projection.min()>=-.0005 and projection.max()<=length+.0005)),
        minimum150mm_tip_clearance=report['minimum_modeled_tip_height_above_measured_brick_m']>=.150,
        bounded_cartesian_rotation=bool(joint_space or report['maximum_orientation_step_deg']<=15.),
        stock_hand_cloud_clear=clear,full_arm_cloud_clear=report['full_arm_observed_cloud_clear'],full_arm_table_clear=report['full_arm_table_clear'],camera_bracket_assumed_envelope_clear=envelope_clear)
    report['checks']=checks;report['failed_checks']=[name for name,passed in checks.items()if not passed]
    report['passed_numeric_checks']=all(checks.values())
    report['path_mode']='joint_space' if joint_space else 'cartesian'
    if joint_space:report['limits_of_review']+=' Joint-space mode has no straight-line/orientation-step restriction; full-arm/table collision review is separately required.'
    if include_dense_poses:report['_dense_tcp_poses']=poses.tolist()
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('plan');p.add_argument('--urdf',required=True);p.add_argument('--snapshot',required=True);p.add_argument('--handeye',required=True);p.add_argument('--geometry',required=True);p.add_argument('--output',required=True);p.add_argument('--joint-space',action='store_true');p.add_argument('--opening-m',type=float,default=.080);a=p.parse_args()
    r=review(a.plan,a.urdf,a.snapshot,a.handeye,a.geometry,a.joint_space,a.opening_m);Path(a.output).write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items()if k not in('full_stock_hand_observed_cloud_sweep','full_arm_convex_hull_review','joint_limits')},indent=2));return 0 if r['passed_numeric_checks'] else 2
if __name__=='__main__':raise SystemExit(main())
