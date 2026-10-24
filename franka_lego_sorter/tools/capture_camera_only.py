"""Read camera without a robot connection; output cannot qualify robot-frame poses."""
import argparse,json
from pathlib import Path
import cv2,numpy as np,yaml
from lego_sorter.camera import RealSenseCamera
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='config/inspect_720.yaml');p.add_argument('--output',required=True);a=p.parse_args()
c=yaml.safe_load(Path(a.config).read_text())
with RealSenseCamera(c['camera']) as camera:f=camera.capture_burst()
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(out,color_bgr=f.color_bgr,depth_m=f.depth_m,intrinsics_json=json.dumps(f.intrinsics),metadata_json=json.dumps(f.metadata),captured_monotonic=f.captured_monotonic,robot_pose_available=False)
if not cv2.imwrite(str(out.with_suffix('.png')),f.color_bgr):raise RuntimeError('Cannot save image')
report={'camera_only':True,'robot_pose_available':False,'robot_connection_opened':False,'not_for_robot_frame_geometry':True,'camera':f.metadata}
out.with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
