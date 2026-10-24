#!/usr/bin/env python3
"""Read-only measurement of explicitly traced template faces; no hardware access."""
import argparse,sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np
from lego_sorter.camera import Frame
from lego_sorter.pallet import fit_surface_plane
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('snapshot',type=Path);parser.add_argument('polygons',type=Path);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();cv2.setNumThreads(1);p=args.snapshot
with np.load(p,allow_pickle=False) as a:f=Frame(a['color_bgr'],a['depth_m'],json.loads(str(a['intrinsics_json'])),float(a['captured_monotonic']),json.loads(str(a['metadata_json'])))
source=json.loads(args.polygons.read_text())
if source['snapshot_sha256']!=hashlib.sha256(p.read_bytes()).hexdigest():raise ValueError('Tracedface snapshot hash mismatch; reobserve changed scene')
polys={item['slot_id']:item['observed_face_polygon_px'] for item in source['slots']}
if len(polys)!=12:raise ValueError('Exactly12 independently traced actual face polygons required')
image=f.color_bgr.copy()
from lego_sorter.template_registration import group_measured_slots
items=[];maskall=np.zeros(f.depth_m.shape,np.uint8)
for sid,poly in polys.items():
 mask=np.zeros(f.depth_m.shape,np.uint8);cv2.fillPoly(mask,[np.array(poly)],255);interior=cv2.erode(mask,np.ones((7,7),np.uint8))>0;valid=interior&(f.depth_m>0)&np.isfinite(f.depth_m);point,normal,rms=fit_surface_plane(f,valid)
 def intersect(pixel):
  ray=f.deproject(*map(float,pixel),1.);return ray*float(normal@point)/float(normal@ray)
 corners=np.array([intersect(p) for p in poly]);edges=[((corners[(i+1)%4]-corners[i])+(corners[(i+2)%4]-corners[(i+3)%4]))/2 for i in range(2)];lengths=[np.linalg.norm(e) for e in edges];axis=edges[int(np.argmax(lengths))];axis/=np.linalg.norm(axis)
 center=np.array(poly).mean(axis=0);item=dict(slot_id=sid,point_camera_m=intersect(center).tolist(),long_axis_camera=axis.tolist(),surface_normal_camera=normal.tolist(),length_m=float(max(lengths)),width_m=float(min(lengths)),center_px=center.tolist(),box_px=poly,surface_plane_rms_m=rms,valid_fraction=float(valid.sum()/interior.sum()));items.append(item)
 cv2.polylines(image,[np.array(poly)],True,(0,255,0),1);cv2.putText(image,sid,tuple(center.astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.32,(255,255,255),1)
try:registration=group_measured_slots(items,dict(template_type='brick_layer',template_type_verified=False)).to_dict();failure=None
except ValueError as e:registration=None;failure=str(e)
report=dict(snapshot=str(p),snapshot_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),observed_faces=12,observed_slots=items,registration=registration,failure=failure,execution_eligible=False,plane_support='Each actual traced face eroded7px to exclude sidewall/background edge-mixing; original observed corners intersect that independently fitted plane. Same3mm RMS gate unchanged.',surface_reference='visible_RGBD_top_only_body_stud_unverified',layer_pitch_measured=False,layer_count_verified=False,notes='No inferred centers/corners/spacing, no nominal studheight subtraction. Local fit scatter not absolute depth accuracy.')
out=args.output.with_suffix('.json');out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));cv2.imwrite(str(out.with_suffix('.png')),image);print(json.dumps(dict(failure=failure,slot_count=len(items),rms_range_m=[min(i['surface_plane_rms_m'] for i in items),max(i['surface_plane_rms_m'] for i in items)],dimensions_m=[[i['slot_id'],i['length_m'],i['width_m']] for i in items],registration_metrics=None if registration is None else registration['metrics']),indent=2))
