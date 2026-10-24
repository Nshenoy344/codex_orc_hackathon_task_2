#!/usr/bin/env python3
"""Offline observed-table relocation suggestions; no hardware or automatic trial."""
import json
from pathlib import Path
import numpy as np
import cv2,yaml
from lego_sorter.isolated_brick import load_frame,_cloud,_hand_hull,_full_hand_obstructions
from lego_sorter.charuco_observation import fit_depth_plane,plane_points
from lego_sorter.pallet_runner import tcp_at_surface

root=Path(__file__).resolve().parents[1];folder=root/'data/commissioning'
snapshot=folder/'isolated_720_restored_view.npz'
if not snapshot.exists():snapshot=folder/'isolated_720_view.npz'
f,G=load_frame(snapshot)
X=np.array(yaml.safe_load((folder/'charuco_rgbd_plane_candidate.yaml').read_text())['T_tcp_color']);B=G@X
old=json.loads((folder/'isolated_trial_geometry.json').read_text())
R=np.array(old['poses']['grasp'])[:3,:3]
hull=_hand_hull('/home/nshenoy/franka_ros2_ws/src/franka_description/meshes/robot_ee/franka_hand_white/collision/hand.stl')
cloud,px=_cloud(f);base=cloud@B[:3,:3].T+B[:3,3]
results=[]
for center in [(75,75),(100,75),(150,75),(50,50),(100,120),(200,75),(350,75),(120,200)]:
    u,v=center;pad=30
    table_poly=np.array([[u-pad,v-pad],[u+pad,v-pad],[u+pad,v+pad],[u-pad,v+pad]])
    try:
        n,d,quality=fit_depth_plane(f,table_poly)
        if n[2]>0:n=-n;d=-d
        nb=B[:3,:3]@n
        if nb[2]<.98:raise ValueError('Not a horizontal measured tabletop patch')
        tablecam=plane_points(f,np.array([center]),n,d)[0];tablebase=(B@np.r_[tablecam,1])[:3]
        visible=tablebase.copy();visible[2]+=old['measured_top_height_over_table_m']/nb[2]
        body=visible-np.array([0,0,.0018]);orient=np.eye(4);orient[:3,:3]=R
        grasp=tcp_at_surface(body,orient,.0095,.003)
        stages={}
        for height in (0.,.020,.050,.100):
            pose=grasp.copy();pose[2,3]+=height
            stages[str(height)]=_full_hand_obstructions(base,pose,hull,opening=.030,margin=.002)
        # Separately test measured tabletop support in a minimum30mm neighborhood.
        inside=(np.linalg.norm(base[:,:2]-tablebase[:2],axis=1)<.030)
        heights=(cloud@n+d)[inside]
        clutter=np.count_nonzero(heights>.003)
        qualified=all(not v['palm_collision_model_points']and not v['finger_collision_model_points']for v in stages.values()) and clutter<=max(10,int(inside.sum()*.01))
        results.append(dict(pixel_center=list(center),measured_table_camera_m=tablecam.tolist(),measured_table_base_m=tablebase.tolist(),
            table_normal_base=nb.tolist(),table_plane_quality=quality,
            future_grasp_pose=grasp.tolist(),future_hover100_pose=grasp.copy().tolist(),
            surrounding_table_cloud_samples=int(inside.sum()),surrounding_raised_point_count=int(clutter),
            stock_hand_cloud_checks=stages,observed_geometry_clear=bool(qualified),
            assumptions='Same test brick measured9.15mm visible topheight, nominalstud1.8mm subtraction onlydiagnostic; no placedobject or grasp hasbeenverified.'))
        results[-1]['future_hover100_pose'][2][3]+=.100
    except ValueError as e:results.append(dict(pixel_center=list(center),observed_geometry_clear=False,reason=str(e)))
selected=next((x for x in results if x.get('observed_geometry_clear')and x['measured_table_base_m'][0]>.30),None)
report=dict(snapshot=str(snapshot),suggestions=results,selected=selected,
    limitations='Measured observed cloud clearance only, not unseen-space, camera-bracket or IK certification; operator must place brick and a fresh independent capture must requalify before descent.')
(folder/'isolated_relocation_suggestions.json').write_text(json.dumps(report,indent=2))
image=f.color_bgr.copy()
if selected:
    center=tuple(selected['pixel_center']);cv2.rectangle(image,(center[0]-30,center[1]-30),(center[0]+30,center[1]+30),(0,220,0),3)
    cv2.putText(image,'PLACE TEST BRICK HERE', (center[0]+35,center[1]),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,180,0),2)
cv2.imwrite(str(folder/'isolated_relocation_suggestion.png'),image)
print(json.dumps(report,indent=2))
