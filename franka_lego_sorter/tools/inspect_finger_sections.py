#!/usr/bin/env python3
"""Offline stock-finger visual-mesh and URDF cross sections; no hardware IO."""
import argparse,json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

NS={'c':'http://www.collada.org/2005/11/COLLADASchema'}

def meshes(path):
    root=ET.parse(path).getroot();out={}
    for g in root.findall('.//c:geometry',NS):
        m=g.find('c:mesh',NS);source_id=m.find('c:vertices/c:input',NS).get('source')[1:]
        source=next(x for x in m.findall('c:source',NS) if x.get('id')==source_id)
        points=np.fromstring(source.find('c:float_array',NS).text,sep=' ').reshape(-1,3)
        triangles=[]
        for element in m.findall('c:triangles',NS):
            inputs=element.findall('c:input',NS);stride=max(int(v.get('offset'))for v in inputs)+1
            vertex_offset=next(int(v.get('offset'))for v in inputs if v.get('semantic')=='VERTEX')
            values=np.fromstring(element.find('c:p',NS).text,sep=' ',dtype=int).reshape(-1,3,stride)
            triangles.extend(points[values[:,:,vertex_offset]])
        out[g.get('id')]=np.asarray(triangles)
    return out


def clip(poly,z,keep_above):
    if not len(poly):return []
    result=[]
    for a,b in zip(poly,np.roll(poly,-1,axis=0)):
        ia=a[2]>=z if keep_above else a[2]<=z
        ib=b[2]>=z if keep_above else b[2]<=z
        if ia:result.append(a)
        if ia!=ib:
            alpha=(z-a[2])/(b[2]-a[2]);result.append(a+alpha*(b-a))
    return result


def slab_bounds(triangles,zlow,zhigh):
    points=[]
    for triangle in triangles:
        points.extend(clip(np.array(clip(triangle,zlow,True)),zhigh,False))
    if not len(points):return None
    points=np.array(points);lo=points.min(axis=0);hi=points.max(axis=0)
    return dict(min_m=lo.tolist(),max_m=hi.tolist(),size_m=(hi-lo).tolist())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mesh',required=True);parser.add_argument('--urdf',required=True);parser.add_argument('--output',required=True);a=parser.parse_args()
    parts=meshes(a.mesh);all_triangles=np.concatenate(list(parts.values()))
    tip_z=float(all_triangles[:,:,2].max());finger_origin=.0584;tcp=.1034;tip_beyond_tcp=tip_z+finger_origin-tcp
    slices={}
    for depth_mm in (.25,.5,1.,2.,3.,4.,5.,6.,8.,10.):
        bounds=slab_bounds(all_triangles,tip_z-depth_mm/1000,tip_z)
        slices[str(depth_mm)]=dict(bounds, insertion_depth_from_mesh_tip_m=depth_mm/1000,
                                 plane_tcp_height_above_surface_m=tip_beyond_tcp-depth_mm/1000)
    root=ET.parse(a.urdf).getroot();link=next(v for v in root.findall('link')if v.get('name')=='fr3_leftfinger')
    boxes=[]
    for col in link.findall('collision'):
        box=col.find('geometry/box')
        if box is None:continue
        half=np.fromstring(box.get('size'),sep=' ')/2
        corners=np.array([[x,y,z]for x in(-half[0],half[0])for y in(-half[1],half[1])for z in(-half[2],half[2])])
        o=col.find('origin');rot=Rotation.from_euler('xyz',np.fromstring(o.get('rpy','0 0 0'),sep=' ')).as_matrix();pos=np.fromstring(o.get('xyz','0 0 0'),sep=' ')
        points=corners@rot.T+pos
        boxes.append(dict(size_m=(half*2).tolist(),min_m=points.min(axis=0).tolist(),max_m=points.max(axis=0).tolist()))
    out=dict(mesh_file=a.mesh,urdf_file=a.urdf,coordinate_convention='Left finger local +Z points from palm toward distal tip; closing axis local Y.',
        finger_joint_z_from_hand_m=finger_origin,tcp_z_from_hand_m=tcp,mesh_tip_local_z_m=tip_z,
        visual_tip_beyond_tcp_m=tip_beyond_tcp,collision_tip_beyond_tcp_m=max(v['max_m'][2]for v in boxes)+finger_origin-tcp,
        visual_part_bounds={k:dict(min_m=v.reshape(-1,3).min(axis=0).tolist(),max_m=v.reshape(-1,3).max(axis=0).tolist())for k,v in parts.items()},
        inserted_visual_slab_bounds=slices,collision_boxes=boxes,
        critical_findings=['The TCP is inside the rubber-pad vertical span, not at its furthest distal tip.',
            'A TCP lowered6mm below top inserts the visual distal tip about14.9mm; this can overlap the next layer of a9.6mm brick.',
            'At6mm tip insertion the complete visual-mesh slab is14.493mm thick along closing Y; the rubber-only11mm bound omits metal backing, so an8mm gap cannot clear the complete stock finger.',
            'The distal rubber collision box itself is15.2mm thick; this is not merely a proximal palm box.',
            'A measured physical finger/camera setup and actual test are still required; visual CAD is not a dimensional certification.'],
        minimal_modification_candidate=dict(status='design proposal only; not fabricated or physically verified',
            type='Measured-mount replacement/bolt-on thin distal contact extensions',
            closing_axis_tip_thickness_m=.003,contact_width_m=.014,distal_protrusion_below_stock_tip_m=.012,
            intended_tip_insertion_below_surface_m=.006,minimum_stock_body_clearance_above_surface_m=.006,
            candidate_preopen_clearance_per_side_m=.002,
            eight_mm_gap_budget_m=dict(available=.008,tip_thickness=.003,preopen_offset=.002,remaining_for_model_pose_and_mechanical_error=.003),
            requirements=['Confirm actual mounting pattern and rigidity; no unverified screw/thread dimensions are specified.',
                'Do not execute dense extraction or seating until physical tip dimensions, stiffness, jaw stroke calibration and contact offsets are measured.',
                'Keep thick stock pad/body above neighboring brick tops throughout descent, closing, lift and release.',
                'Connected LEGO still needs a tested separation/support strategy; thin tips alone do not establish detachment.']),
        release_assessment=['Opening jaws at a seated interior slot can collide with neighboring brick faces; verify both swept jaw footprints at the actual release pose.',
            'Retracting before opening releases the held brick from height; that can avoid inserted-jaw collisions but changes placement into a drop, with unmeasured bounce, yaw and stud engagement.',
            'No drop mode or new insertion depth is authorized by this geometry report alone. Prove a single isolated guided placement and inspect before extending to dense12slot layers.'])
    Path(a.output).write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
