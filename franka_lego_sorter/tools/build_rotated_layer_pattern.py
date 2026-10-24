"""Offline rotate the entire observed12-slot layer90degrees about its centroid."""
from pathlib import Path
import json,hashlib
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
P=Path('data/commissioning')
def load(n):return json.loads((P/n).read_text())
reg=load('template_reference_uncovered_interior_registration.json');prior=load('first_loose_placement_target.json');B=np.array(prior['T_base_color_reference']);n=np.array(prior['support_plane_base']['normal']);n/=np.linalg.norm(n)
slots=reg['observed_slots'];centers=np.array([B[:3,:3]@np.array(s['point_camera_m'])+B[:3,3]for s in slots]);origin=centers.mean(0);Q=Rotation.from_rotvec(n*np.pi/2).as_matrix();e1=centers[-1]-centers[0];e1-=n*(n@e1);e1/=np.linalg.norm(e1);e2=np.cross(n,e1)
# Each measured original footprint is independently represented, never grid-inferred.
footprints=[]
for slot,c in zip(slots,centers):
 a=B[:3,:3]@np.array(slot['long_axis_camera']);a-=n*(n@a);a/=np.linalg.norm(a);cross=np.cross(n,a)
 footprints.append(np.array([c+x*slot['length_m']/2*a+y*slot['width_m']/2*cross for x,y in[(-1,-1),(1,-1),(1,1),(-1,1)]]))
def inside(points,polygon):
 p=np.column_stack(((points-origin)@e1,(points-origin)@e2));v=np.column_stack(((polygon-origin)@e1,(polygon-origin)@e2));cross=[]
 for i in range(4):
  edge=v[(i+1)%4]-v[i];d=p-v[i];cross.append(edge[0]*d[:,1]-edge[1]*d[:,0])
 z=np.array(cross);return np.all(z>=-1e-9,axis=0)|np.all(z<=1e-9,axis=0)
outputs=[]
for slot,c,f in zip(slots,centers,footprints):
 cf=origin+Q@(c-origin);rf=(f-origin)@Q.T+origin
 # Sampling is a coverage diagnostic over measured face polygons, not stud contact.
 u,v=np.meshgrid(np.linspace(-1,1,41),np.linspace(-1,1,21));samples=cf+u.ravel()[:,None]*(rf[1]-rf[0])/2+v.ravel()[:,None]*(rf[3]-rf[0])/2
 covered=np.zeros(len(samples),bool);parts=[]
 for base,poly in zip(slots,footprints):
  hit=inside(samples,poly);covered|=hit
  if np.any(hit):parts.append(base['slot_id'])
 contact=np.column_stack(((samples[covered]-origin)@e1,(samples[covered]-origin)@e2)).astype(np.float32)
 hull=cv2.convexHull(contact) if len(contact)>=3 else None
 center2=tuple(float(x)for x in [(cf-origin)@e1,(cf-origin)@e2])
 margin=float(cv2.pointPolygonTest(hull,center2,True))if hull is not None else float('-inf')
 outputs.append(dict(center_inside_contact_hull_margin_m=margin,slot_id=slot['slot_id'],center_base_m=cf.tolist(),footprint_base_m=rf.tolist(),long_axis_base=(Q@(B[:3,:3]@np.array(slot['long_axis_camera']))).tolist(),measured_face_coverage_fraction=float(covered.mean()),overlapping_underlying_slot_ids=parts,support_verified=False))
source_files=[{'path':str((P/name).resolve()),'sha256':hashlib.sha256((P/name).read_bytes()).hexdigest()}for name in['template_reference_uncovered_interior_registration.json','first_loose_placement_target.json']]
out=dict(kind='whole_pattern_90degree_rotation_target_only',rotation_degrees=90,centroid_base_m=origin.tolist(),normal_base=n.tolist(),R_base_pattern_rotation=Q.tolist(),slots=outputs,chosen_first_slot='r3b2',production_ready=False,action_sent=False,layer_pitch_measured=False,support_semantics='Coverage against actual measured old-layer face footprints only; not measured studcontact. Bridging gaps may be valid but requires independent contact/stability review.',source_files=source_files)
(P/'rotated_first_layer_pattern.json').write_text(json.dumps(out,indent=2));print([(s['slot_id'],s['measured_face_coverage_fraction'],s['overlapping_underlying_slot_ids'])for s in outputs])
# Conditional target under the ORIGINAL held-object transform only. A new grasp
# must supply a newly verified object-in-TCP transform before this can be used.
for chosen in ('r3b2','r1b2'):
 s=next(x for x in outputs if x['slot_id']==chosen)
 originalslot=next(x for x in slots if x['slot_id']==chosen)
 oldslotbase=B[:3,:3]@np.array(originalslot['point_camera_m'])+B[:3,3]
 # Reconstruct source object-center offset from the previous target.
 oldT=np.array(prior['T_base_tcp_release']);heldcenter=np.array(prior['source_center_tcp_estimate_m'])
 R=Q@oldT[:3,:3];newcenter=np.array(s['center_base_m']);p=newcenter-R@heldcenter
 env=load('template_upper_envelope_review.json');table=B[:3,:3]@np.array(env['table_point_camera_m'])+B[:3,3]
 overlapping=s['overlapping_underlying_slot_ids'];upper=max(x['height_percentiles_m']['p99']for x in env['samples']if x['slot_id']in overlapping)
 observedmax=max(x['height_percentiles_m']['max']for x in env['samples']if x['slot_id']in overlapping)
 bottom=np.array(prior['estimated_bottom_footprint_tcp_m']);h=float(n@table+upper)
 p[2]=(h+.003-min((bottom@R.T)@n)-n[:2]@p[:2])/n[2]
 T=np.eye(4);T[:3,:3]=R;T[:3,3]=p;A=T.copy();A[2,3]+=.03
 report=dict(kind='rotated_layer_conditional_target_only',slot_id=chosen,whole_pattern_artifact=str((P/'rotated_first_layer_pattern.json').resolve()),T_base_tcp_release=T.tolist(),T_base_tcp_approach30=A.tolist(),estimated_bottom_footprint_base_m=(bottom@R.T+p).tolist(),original_held_transform_only=True,new_grasp_requires_rederived_transform=True,production_ready=False,action_sent=False,center_inside_contact_hull_margin_m=s['center_inside_contact_hull_margin_m'],minimum_gap_above_underlying_p99_m=.003,minimum_gap_above_underlying_observedmax_m=.003+upper-observedmax,support_verified=False,sources=source_files)
 (P/f'rotated_first_layer_{chosen}_conditional_target.json').write_text(json.dumps(report,indent=2))
