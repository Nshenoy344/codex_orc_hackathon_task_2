"""Offline isolated-brick point-trial geometry. Never commands hardware.

The nominal stud height is an explicit diagnostic assumption, not a measured
body-top qualification. Production pallet picking still requires measured body
reference, calibrated stations, and verified individual transfer behavior.
"""
import argparse,json,struct
from pathlib import Path
import cv2
import numpy as np
import yaml
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from .camera import Frame
from .charuco_observation import fit_depth_plane,plane_points
from .geometry import matrix
from .pallet_runner import tcp_at_surface


def load_frame(path):
    a=np.load(path,allow_pickle=False)
    for key,bound in [('robot_drift_m',.001),('robot_drift_rad',.005)]:
        if key not in a or not np.isfinite(float(a[key])) or float(a[key])>bound:
            raise ValueError('Isolated trial requires a stationary measured robot/camera snapshot')
    f=Frame(a['color_bgr'],a['depth_m'],json.loads(str(a['intrinsics_json'])),float(a['captured_monotonic']),json.loads(str(a['metadata_json'])))
    return f,matrix(a['T_base_tcp'])


def _cloud(frame,stride=3):
    y,x=np.mgrid[0:frame.depth_m.shape[0]:stride,0:frame.depth_m.shape[1]:stride]
    z=frame.depth_m[y,x];valid=np.isfinite(z)&(z>.1)&(z<2.)
    y,x,z=y[valid],x[valid],z[valid]
    if any(abs(v)>1e-8 for v in frame.intrinsics.get('coeffs',[])):
        points=np.array([frame.deproject(float(u),float(v),float(d))for v,u,d in zip(y,x,z)])
    else:
        k=frame.intrinsics;points=np.c_[(x-k['ppx'])*z/k['fx'],(y-k['ppy'])*z/k['fy'],z]
    return points,np.c_[x,y]


def _hand_hull(path):
    raw=Path(path).read_bytes();n=struct.unpack_from('<I',raw,80)[0]
    if len(raw)!=84+50*n:raise ValueError('Expected the installed binary collision STL')
    dt=np.dtype([('normal','<f4',3),('vertices','<f4',(3,3)),('attribute','<u2')])
    vertices=np.frombuffer(raw,dtype=dt,count=n,offset=84)['vertices'].reshape(-1,3).astype(float)
    return ConvexHull(vertices).equations


def _full_hand_obstructions(points_base,tcp,hand_hull,opening=.030,margin=.002):
    # Stock hand frame shares TCP orientation and lies103.4mm proximal to it.
    local=(points_base-tcp[:3,3])@tcp[:3,:3]
    hand=local+np.array([0.,0.,.1034])
    palm=np.all(hand@hand_hull[:,:3].T+hand_hull[:,3]<=margin,axis=1)
    boxes=[([0,.0185,.011],[.022,.015,.020],0.),
           ([0,.0068,.0022],[.022,.0088,.0038],0.),
           ([0,.0159,.02835],[.0175,.007,.0235],np.pi/6),
           ([0,.00758,.04525],[.0175,.0152,.0185],0.)]
    finger=np.zeros(len(hand),bool)
    for side in (-1,1):
        q=hand-np.array([0.,side*opening/2,.0584])
        if side<0:q=q@np.diag([-1.,-1.,1.])
        for center,size,rx in boxes:
            v=(q-np.array(center))@Rotation.from_rotvec([rx,0,0]).as_matrix()
            finger|=np.all(np.abs(v)<=np.array(size)/2+margin,axis=1)
    return dict(palm_collision_model_points=int(palm.sum()),finger_collision_model_points=int(finger.sum()),
                nearest_model_clearance_unverified=True)


def inspect_isolated(frame,T_base_tcp,T_tcp_color,roi,table_polygon,hand_mesh,
                     nominal_stud_height_m=.0018,deep_insertion_m=.003,tip_extension_m=.0095,hover_only=False):
    T_base_color=matrix(T_base_tcp)@matrix(T_tcp_color)
    roi=list(map(int,roi));x,y,w,h=roi
    if x<0 or y<0 or w<20 or h<20 or x+w>frame.depth_m.shape[1] or y+h>frame.depth_m.shape[0]:raise ValueError('Invalid complete isolated-brick ROI')
    n,d,table=fit_depth_plane(frame,np.array(table_polygon,float))
    if n[2]>0:n=-n;d=-d
    if table['rms_m']>.0015:raise ValueError('Separate tabletop depth is insufficiently planar')
    normal_base=T_base_color[:3,:3]@n
    if normal_base[2]<.98:raise ValueError('Measured support is not approximately a horizontal tabletop')
    hsv=cv2.cvtColor(frame.color_bgr[y:y+h,x:x+w],cv2.COLOR_BGR2HSV)
    mask=((hsv[:,:,1]>100)&(hsv[:,:,2]<180)).astype(np.uint8)*255
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:raise ValueError('No isolated colored face in ROI')
    contour=max(contours,key=cv2.contourArea)+np.array([[[x,y]]])
    rect=cv2.minAreaRect(contour);box=cv2.boxPoints(rect)
    if np.any(box[:,0]<x+2)or np.any(box[:,0]>x+w-2)or np.any(box[:,1]<y+2)or np.any(box[:,1]>y+h-2):raise ValueError('Isolated face is clipped by its ROI')
    interior=np.zeros(frame.depth_m.shape,np.uint8);cv2.fillConvexPoly(interior,np.round(box).astype(int),255)
    interior=cv2.erode(interior,np.ones((7,7),np.uint8))
    points,pixels=_cloud(frame,stride=1)
    inside=interior[pixels[:,1],pixels[:,0]]>0
    face=points[inside];heights=face@n+d
    plausible=(heights>.004)&(heights<.018)
    if plausible.sum()<200 or plausible.mean()<.85:raise ValueError('ROI does not have reliable one-brick raised support')
    top_height=float(np.percentile(heights[plausible],90))
    height_qualified=.007<=top_height<=.014
    if not height_qualified and not hover_only:raise ValueError('Visible top height is inconsistent with an isolated single brick')
    top_camera=plane_points(frame,np.array([rect[0]]),n,d-top_height)[0]
    corners=plane_points(frame,box,n,d-top_height)
    edges=np.roll(corners,-1,axis=0)-corners;i=int(np.argmax(np.linalg.norm(edges,axis=1)))
    long_axis=edges[i]/np.linalg.norm(edges[i]);dimensions=sorted([float(np.linalg.norm(edges[0])),float(np.linalg.norm(edges[1]))])
    if abs(dimensions[1]-.0318)>.006 or abs(dimensions[0]-.0158)>.005:raise ValueError('Observed footprint is inconsistent with nominal2x4; do not assume a different grasp width')
    visible=(T_base_color@np.r_[top_camera,1])[:3]
    long_base=T_base_color[:3,:3]@long_axis;long_base[2]=0;long_base/=np.linalg.norm(long_base)
    closing=np.array([-long_base[1],long_base[0],0.])
    orientation=np.eye(4);orientation[:3,:3]=np.c_[np.cross(closing,[0,0,-1.]),closing,[0,0,-1.]]
    if not 0<nominal_stud_height_m<.003 or not .002<deep_insertion_m<=.003:raise ValueError('Only explicit diagnostic3mm body insertion with declared nominalstud assumption is allowed')
    body_assumption=visible-np.array([0,0,nominal_stud_height_m])
    grasp=tcp_at_surface(body_assumption,orientation,tip_extension_m,deep_insertion_m)
    hover=grasp.copy();hover[2,3]+=.100
    hover200=grasp.copy();hover200[2,3]+=.200
    # First hover deliberately keeps current jaw yaw, not brick alignment.
    keep_closing=np.array([T_base_tcp[0,1],T_base_tcp[1,1],0.]);keep_closing/=np.linalg.norm(keep_closing)
    first_hover=hover200.copy();first_hover[:3,:3]=np.c_[np.cross(keep_closing,[0,0,-1.]),keep_closing,[0,0,-1.]]
    approach=grasp.copy();approach[2,3]+=.020
    lift=grasp.copy();lift[2,3]+=.010
    tip=grasp[:3,3]+grasp[:3,:3]@np.array([0,0,tip_extension_m])
    table_point_camera=plane_points(frame,np.array([rect[0]]),n,d)[0]
    table_base=(T_base_color@np.r_[table_point_camera,1])[:3]
    tip_table_clearance=float((tip-table_base)@normal_base)
    cloud,px=_cloud(frame,stride=3);base=cloud@T_base_color[:3,:3].T+T_base_color[:3,3]
    excluded=interior[px[:,1],px[:,0]]>0;surroundings=base[~excluded]
    hull=_hand_hull(hand_mesh)
    sweep=[]
    for height in (.200,.100,.075,.050,.025,.020,.010,0.):
        p=grasp.copy();p[2,3]+=height
        sweep.append(dict(tcp_above_grasp_m=height,**_full_hand_obstructions(surroundings,p,hull)))
    #30mm full free ring is diagnostic; actual model sweep remains authoritative.
    scale=(frame.intrinsics['fx']+frame.intrinsics['fy'])/2/float(top_camera[2])
    ring_pad=.030*scale
    ring_visible=bool(box[:,0].min()-ring_pad>=0 and box[:,0].max()+ring_pad<frame.depth_m.shape[1]and box[:,1].min()-ring_pad>=0 and box[:,1].max()+ring_pad<frame.depth_m.shape[0])
    blockers=[]
    if not height_qualified:blockers.append('Measuredtopheight outside7..14mm singlebrickacceptance; hoveronly, reobserve before contact')
    if tip_table_clearance<.003: blockers.append('Predictedtip/table clearance is under3mm before uncertainty')
    if not ring_visible:blockers.append('Full30mm free ring is not observed; use exact swept-volume checks and a centered reobservation')
    if any(s['palm_collision_model_points']or s['finger_collision_model_points']for s in sweep):blockers.append('Observed surrounding cloud intersects the full stockHand descent/opening collision model')
    visibility={}
    corner_base=corners@T_base_color[:3,:3].T+T_base_color[:3,3]
    for name,pose in [('hover100',hover),('hover200',hover200),('first_hover200',first_hover)]:
        camera_pose=pose@matrix(T_tcp_color)
        cp=(corner_base-camera_pose[:3,3])@camera_pose[:3,:3]
        intr=frame.intrinsics
        projected=np.c_[intr['fx']*cp[:,0]/cp[:,2]+intr['ppx'],intr['fy']*cp[:,1]/cp[:,2]+intr['ppy']]
        face_visible=bool(np.all(cp[:,2]>.18) and np.all(projected[:,0]>10) and np.all(projected[:,1]>10) and np.all(projected[:,0]<frame.depth_m.shape[1]-10) and np.all(projected[:,1]<frame.depth_m.shape[0]-10))
        visibility[name]=dict(brick_corners_px=projected.tolist(),minimum_optical_depth_m=float(cp[:,2].min()),complete_face_visible_with10pxmargin=face_visible)
    return dict(kind='isolated_brick_diagnostic_geometry_only',physical_commissioning_verified=False,
        production_source_pick_qualified=False,point_trial_geometry_qualified=not blockers,
        roi_px=roi,brick_box_px=box.tolist(),visible_top_camera_m=top_camera.tolist(),visible_top_base_m=visible.tolist(),
        measured_footprint_short_long_m=dimensions,measured_top_height_over_table_m=top_height,
        measured_table_plane=dict(normal_camera=n.tolist(),offset_camera_m=float(d),normal_base=normal_base.tolist(),**table),
        table_point_base_m=table_base.tolist(),predicted_tip_table_clearance_m=tip_table_clearance,
        explicit_nominal_assumptions=dict(stud_height_m=nominal_stud_height_m,brick_grasp_width_m=.0158,reference='90thpercentile visible face; studheight subtraction is a diagnostic assumption, NOT measured production bodytop'),
        proposed_gripper=dict(preopen_width_m=.030,grasp_width_m=.0158,force_n=15.,tip_extension_model_m=tip_extension_m),
        hover_projection=visibility,
        poses=dict(first_hover200=first_hover.tolist(),hover200=hover200.tolist(),hover100=hover.tolist(),approach20=approach.tolist(),grasp=grasp.tolist(),test_lift10=lift.tolist(),return_before_release=grasp.tolist()),
        full_hand_observed_cloud_sweep=sweep,full30mm_ring_visible=ring_visible,blockers=blockers,
        workflow='Plan/review hover only first; do not execute descent/grasp/release while blockers remain. Later each stage must be separately planned from fresh measured state, verified, logged and reviewed; no automatic full-trial action is provided.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('snapshot');p.add_argument('--handeye',required=True);p.add_argument('--roi',nargs=4,type=int,required=True);p.add_argument('--table-polygon',nargs=8,type=float,required=True);p.add_argument('--hand-mesh',required=True);p.add_argument('--output',required=True);p.add_argument('--hover-only',action='store_true')
    a=p.parse_args();frame,g=load_frame(a.snapshot);candidate=yaml.safe_load(Path(a.handeye).read_text())
    if candidate.get('diagnostics',{}).get('accepted')is not True:raise ValueError('Numerically rejected handeye cannot be used even for isolated commissioning')
    result=inspect_isolated(frame,g,matrix(candidate['T_tcp_color']),a.roi,np.array(a.table_polygon).reshape(4,2),a.hand_mesh,hover_only=a.hover_only)
    result.update(snapshot=str(Path(a.snapshot).resolve()),handeye_artifact=str(Path(a.handeye).resolve()),handeye_physical_verified=candidate.get('physical_commissioning_verified')is True)
    Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items()if k!='poses'},indent=2))
if __name__=='__main__':main()
