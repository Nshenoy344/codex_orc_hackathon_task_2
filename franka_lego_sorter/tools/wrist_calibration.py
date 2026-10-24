"""Read-only two-camera calibration preview/collector. No motion interfaces."""
import argparse,hashlib,json,subprocess,time,threading
from pathlib import Path
import cv2,numpy as np,yaml
from lego_sorter.camera import RealSenseCamera,limit_vision_threads
from lego_sorter.wrist_stereo import detect_corners,solve

ROOT=Path(__file__).resolve().parents[1]
DEVICE='/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._HD_Pro_Webcam_C920_HD_Pro_Webcam_C920-video-index0'

def robot_snapshot():
    # Refuse competing FCI readers/controllers. The helper itself calls readOnce only.
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            try:name=(entry/'comm').read_text().strip()
            except (OSError,PermissionError):continue
            if name in ('ros2_control_no','ros2_control_node'):
                return json.loads(subprocess.check_output(['bash',str(ROOT/'tools/read_ros_snapshot.sh')],timeout=10))
    before=time.monotonic();s=json.loads(subprocess.check_output([str(ROOT/'tools/read_robot_snapshot')],timeout=5))
    s['host_monotonic_before']=before;s['host_monotonic_after']=time.monotonic()
    return s

class Webcam:
    def __init__(self,device):
        self.cap=cv2.VideoCapture(device,cv2.CAP_V4L2)
        if not self.cap.isOpened():raise RuntimeError('Cannot open wrist webcam')
        self.cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'));self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280);self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720);self.cap.set(cv2.CAP_PROP_FPS,30);self.cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        self.running=True;self.latest=None;self.lock=threading.Lock();self.thread=threading.Thread(target=self.read,daemon=True);self.thread.start()
    def read(self):
        while self.running:
            ok,im=self.cap.read()
            if ok:
                with self.lock:self.latest=(time.monotonic(),im)
            else:time.sleep(.02)
    def frame(self):
        with self.lock:
            if self.latest is None:raise RuntimeError('Waiting for webcam')
            stamp,im=self.latest;return stamp,im.copy()
    def close(self):self.running=False;self.thread.join(timeout=2);self.cap.release()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',default='data/wrist_calibration');p.add_argument('--device',default=DEVICE);p.add_argument('--auto',action='store_true');p.add_argument('--capture-once',action='store_true');p.add_argument('--solve',action='store_true');p.add_argument('--timeout',type=float,default=25.);a=p.parse_args()
    limit_vision_threads();folder=Path(a.session);folder.mkdir(parents=True,exist_ok=True)
    board_path=ROOT/'data/commissioning/charuco_reference_board.yaml';board_doc=yaml.safe_load(board_path.read_text());board=board_doc['board_config'];hand_path=ROOT/'data/commissioning/charuco_rgbd_plane_candidate.yaml';hand=yaml.safe_load(hand_path.read_text())
    if a.solve:
        samples=[json.loads(f.read_text())for f in sorted(folder.glob('sample_*.json'))]
        if not samples:raise RuntimeError('No calibration samples yet')
        first=np.load(samples[0]['rgbd_file']);intr=json.loads(str(first['intrinsics_json']));K=np.array([[intr['fx'],0,intr['ppx']],[0,intr['fy'],intr['ppy']],[0,0,1.]])
        signatures={json.dumps(s['webcam_settings'],sort_keys=True)for s in samples}
        if len(signatures)!=1:raise RuntimeError('Webcam focus/zoom/resolution changed between samples')
        result=solve(samples,board,samples[0]['webcam_size'],K,np.array(intr.get('coeffs',[0]*5)),hand['T_tcp_color'],first['T_color_depth'])
        result.update(board_config=board,board_scale_provenance=board_doc.get('scale_provenance'),handeye_source=str(hand_path),handeye_sha256=hashlib.sha256(hand_path.read_bytes()).hexdigest(),webcam_device=a.device,webcam_settings=samples[0]['webcam_settings'],sample_files=[dict(path=str(f.resolve()),sha256=hashlib.sha256(f.read_bytes()).hexdigest())for f in sorted(folder.glob('sample_*.json'))])
        (folder/'calibration.yaml').write_text(yaml.safe_dump(result,sort_keys=False));print(json.dumps(result,indent=2));return
    web=Webcam(a.device);saved=[{int(i):np.array(v)for i,v in json.loads(p.read_text())['webcam_corners'].items()}for p in sorted(folder.glob('sample_*.json'))];last_capture=0.;previous=None;stable_since=time.monotonic();status='Place board in BOTH views';failures=0
    config=yaml.safe_load((ROOT/'config/inspect_720.yaml').read_text())['camera'];config['warmup_frames']=15
    try:
      with RealSenseCamera(config)as rs:
        begun=time.monotonic()
        while True:
          if a.capture_once and time.monotonic()-begun>a.timeout:raise RuntimeError('Board sample timeout: '+status)
          if (folder/'stop.request').exists():break
          f=rs.capture()
          try:stamp,im=web.frame()
          except RuntimeError:continue
          rc,ri=detect_corners(f.color_bgr,board);wc,wi=detect_corners(im,board);common=sorted(set(rc)&set(wc));now=time.monotonic()
          feature=np.array([wc[i]for i in sorted(wc)])if len(wc)>=10 else None
          comparable=previous is not None and set(wc)==set(previous)
          motion=max((np.linalg.norm(wc[i]-previous[i])for i in wc),default=99.)if comparable else 99.
          if motion>1.:stable_since=now
          previous=wc;ready=len(common)>=10 and now-stable_since>.8 and abs(stamp-f.captured_monotonic)<.15
          if ready:
            # Compare corresponding corners against all saved poses; no repeated stationary samples.
            novel=not saved or all(np.mean([np.linalg.norm(wc[i]-s[i])for i in set(wc)&set(s)])>12 for s in saved if len(set(wc)&set(s))>=6)
          else:novel=False
          title=f'RS corners:{len(rc)}  Logitech:{len(wc)}  common:{len(common)}  saved:{len(list(folder.glob("sample_*.json")))}'
          canvas=np.hstack([cv2.resize(ri,(640,360)),cv2.rotate(cv2.resize(wi,(640,360)),cv2.ROTATE_180)])
          cv2.putText(canvas,title,(10,23),cv2.FONT_HERSHEY_SIMPLEX,.57,(0,255,255),2)
          cv2.putText(canvas,status[:115],(10,48),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1)
          cv2.putText(canvas,'ARM IS NOT CONTROLLED | Tilt/move board | SPACE save | Q quit | focus/zoom must stay fixed',(10,344),cv2.FONT_HERSHEY_SIMPLEX,.47,(255,255,255),1)
          cv2.imwrite(str(folder/'preview.jpg'),canvas)
          key=-1
          if not a.capture_once:cv2.imshow('Wrist camera calibration - READ ONLY',canvas);key=cv2.waitKey(1)&255
          if key==ord('q'):break
          requested=(a.auto and ready and novel and now-last_capture>2.)or key==32 or(a.capture_once and ready)
          if requested and ready:
            try:
              # Freeze the lens for intrinsic calibration. Caller can choose a new
              # focus before a session; it must remain constant for all samples.
              controls=subprocess.check_output(['v4l2-ctl','-d',a.device,'--get-ctrl=focus_automatic_continuous,focus_absolute,zoom_absolute'],text=True)
              settings={line.split(':')[0].strip():int(line.split(':')[1])for line in controls.splitlines()if ':'in line}
              if settings.get('focus_automatic_continuous')!=0:raise RuntimeError('Focus is automatic; lock focus before collecting calibration')
              rb=robot_snapshot();time.sleep(.12);f2=rs.capture();_,im2=web.frame();rc2,_=detect_corners(f2.color_bgr,board);wc2,_=detect_corners(im2,board);ra=robot_snapshot()
              ids=sorted(set(rc)&set(wc)&set(rc2)&set(wc2))
              if len(ids)<10 or max(np.linalg.norm(wc[i]-wc2[i])for i in ids)>1. or max(np.linalg.norm(rc[i]-rc2[i])for i in ids)>1.:raise RuntimeError('Board moved during sample; hold steady')
              T0=np.array(rb['O_T_EE']).reshape(4,4,order='F');T1=np.array(ra['O_T_EE']).reshape(4,4,order='F')
              if np.linalg.norm(T0[:3,3]-T1[:3,3])>.001 or np.arccos(np.clip((np.trace(T0[:3,:3].T@T1[:3,:3])-1)/2,-1,1))>.005:raise RuntimeError('Robot moved during sample')
              prefix=folder/f'sample_{time.time_ns()}';rgbd=prefix.with_suffix('.npz');png=prefix.with_suffix('.png')
              np.savez_compressed(rgbd,color_bgr=f2.color_bgr,depth_m=f2.depth_m,intrinsics_json=json.dumps(f2.intrinsics),metadata_json=json.dumps(f2.metadata),captured_monotonic=f2.captured_monotonic,T_base_tcp=T1,T_color_depth=rs.T_color_depth,robot_drift_m=float(np.linalg.norm(T0[:3,3]-T1[:3,3])),robot_drift_rad=float(np.arccos(np.clip((np.trace(T0[:3,:3].T@T1[:3,:3])-1)/2,-1,1))))
              cv2.imwrite(str(png),im2)
              record=dict(rs_corners={str(i):rc2[i].tolist()for i in rc2},webcam_corners={str(i):wc2[i].tolist()for i in wc2},rgbd_file=str(rgbd.resolve()),webcam_file=str(png.resolve()),webcam_size=[im2.shape[1],im2.shape[0]],webcam_settings=settings,robot_before=rb,robot_after=ra,T_base_tcp=T1.tolist(),motion_command_sent=False,board_may_move_between_samples=True,board_must_be_stationary_during_sample=True)
              prefix.with_suffix('.json').write_text(json.dumps(record,indent=2));saved.append(wc2);last_capture=now;status='SAVED: move/tilt board to a different position';print(json.dumps(dict(saved=str(prefix),common_corners=len(ids))),flush=True)
              if a.capture_once:break
            except Exception as e:status=str(e);last_capture=now;print(status,flush=True)
          elif len(common)<10:status='Need >=10 common corners: bring the board fully into BOTH images'
          elif not ready:status='Hold the board still briefly'
          elif not novel:status='Tilt board left/right/forward/back or move toward image edges for a distinct view'
    finally:web.close();cv2.destroyAllWindows()

if __name__=='__main__':main()
