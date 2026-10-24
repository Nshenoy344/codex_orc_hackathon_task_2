#!/usr/bin/env python3
"""Offline target only. Explicit conditional source silhouette metric projection."""
import json, hashlib
from pathlib import Path
import numpy as np
import yaml
from lego_sorter.held_bottom_plane import bottom_plane_in_tcp, release_tcp_z

P=Path('data/commissioning')
def read(name): return json.loads((P/name).read_text())
def unit(x): return x/np.linalg.norm(x)
def pin(name):
 p=P/name
 return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
hand=P/'charuco_rgbd_plane_candidate.yaml'
H=np.array(yaml.safe_load(hand.read_text())['T_tcp_color'])
z=np.load(P/'template_reference_uncovered_paired.npz');B=np.array(z['T_base_tcp'])@H
source=np.load(P/'isolated_close_observation.npz');S=np.array(source['T_base_tcp'])@H
s=read('isolated_close_nominal_diagnostic_targets.json')['planes'][0]
g=np.array(read('isolated_first_grasp.json')['T_base_tcp_before'])
ns=unit(np.array(s['measured_table_plane']['normal_base']));floor=np.array(s['floor_ray_point_base_m']);hs=float(ns@floor)
bn,bh=bottom_plane_in_tcp(g,ns,hs)
poly=np.array(read('isolated_close_observation_review.json')['visible_face_polygon_px']);k=json.loads(str(source['intrinsics_json']))
# Existing diagnostic used11.4mm top for perspective only; never use it for bottom Z.
points=[]
for u,v in poly:
 ray=S[:3,:3]@np.array([(u-k['ppx'])/k['fx'],(v-k['ppy'])/k['fy'],1.])
 top=S[:3,3]+ray*((hs+.0114-ns@S[:3,3])/(ns@ray))
 bottom=top-ns*.0114
 points.append(g[:3,:3].T@(bottom-g[:3,3]))
points=np.array(points);center=points.mean(0)
reg=read('template_reference_uncovered_interior_registration.json');slot=next(x for x in reg['observed_slots'] if x['slot_id']=='r3b2')
review=read('template_upper_envelope_review.json');table=np.array(review['table_point_camera_m']);n=B[:3,:3]@np.array(review['normal_camera']);n=unit(n);tablebase=B[:3,:3]@table+B[:3,3]
env=next(x for x in review['samples'] if x['slot_id']=='r3b2')['height_percentiles_m']
h=float(n@tablebase+env['p99']);long=B[:3,:3]@np.array(slot['long_axis_camera']);long=unit(long-n*(n@long))
# Long axis is local TCP X for the actual across-width grasp.
localx=np.array([1.,0,0]);localx=unit(localx-bn*(bn@localx));localy=np.cross(bn,localx)
worldy=np.cross(n,long);R=np.column_stack([long,worldy,n])@np.column_stack([localx,localy,bn]).T
# Choose closest equivalent yaw to the grasp, reducing unnecessary wrist travel.
if np.trace(g[:3,:3].T@R)<np.trace(g[:3,:3].T@(np.column_stack([-long,-worldy,n])@np.column_stack([localx,localy,bn]).T)):
 R=np.column_stack([-long,-worldy,n])@np.column_stack([localx,localy,bn]).T
slotbase=B[:3,:3]@np.array(slot['point_camera_m'])+B[:3,3]
xy=(slotbase-R@center)[:2];tcpz=release_tcp_z(xy,R,bn,bh,n,h,points,.003)
T=np.eye(4);T[:3,:3]=R;T[:3,3]=[*xy,tcpz];A=T.copy();A[2,3]+=.030
worldpoints=points@R.T+T[:3,3]
out={'kind':'first_loose_placement_target_only','slot_id':'r3b2','production_placement_ready':False,'action_sent':False,'T_base_tcp_release':T.tolist(),'T_base_tcp_approach30':A.tolist(),'T_base_color_reference':B.tolist(),'estimated_bottom_footprint_base_m':worldpoints.tolist(),'estimated_bottom_footprint_tcp_m':points.tolist(),'bottom_plane_tcp':{'normal':bn.tolist(),'offset_m':bh},'support_plane_base':{'normal':n.tolist(),'offset_p99_m':h},'minimum_clearance_above_p99_m':float(np.min(worldpoints@n)-h),'minimum_clearance_above_observed_max_m':float(np.min(worldpoints@n)-(h+env['max']-env['p99'])),'source_center_tcp_estimate_m':center.tolist(),'source_xy_assumption':'Observed source silhouette rays projected using prior explicitly diagnostic11.4mm top height; physical grasp succeeded but true object XY center not independently measured','held_bottom_assumptions':['Bottom rested flat on measured source table at closure','Rigid no-slip hold since closure'],'support_semantics':'Actual observed upper-envelope only; no body/stud or snap inference','release_mode':'Loose3mm gap; no force press; open minimally then vertical retract before full opening','required_after_release':'Fresh paired RGBD, measured footprint occupancy/color and signed settled height increment; do not increment production count from target alone','source_files':[pin(x) for x in ['isolated_close_observation.npz','isolated_close_observation_review.json','isolated_close_nominal_diagnostic_targets.json','isolated_first_grasp.json','charuco_rgbd_plane_candidate.yaml','template_reference_uncovered_paired.npz','template_reference_uncovered_interior_registration.json','template_upper_envelope_review.json']]}
(P/'first_loose_placement_target.json').write_text(json.dumps(out,indent=2))
print(json.dumps({k:out[k] for k in ['slot_id','T_base_tcp_release','minimum_clearance_above_p99_m','minimum_clearance_above_observed_max_m']},indent=2))
