"""Capture the identified Logitech C920; no robot commands."""
import argparse,json,time
from pathlib import Path
import cv2
parser=argparse.ArgumentParser()
parser.add_argument('--device',default='/dev/video8')
parser.add_argument('--output',default='data/commissioning/overview.png')
parser.add_argument('--rotation-deg',type=int,choices=(0,90,180,270),default=0,
                    help='Display orientation only; does not establish metric camera extrinsics')
args=parser.parse_args()
camera=cv2.VideoCapture(args.device,cv2.CAP_V4L2)
try:
    if not camera.isOpened():raise RuntimeError('Overview camera unavailable')
    camera.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    for _ in range(15):ok,image=camera.read()
    if not ok:raise RuntimeError('No overview image')
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    if args.rotation_deg:
        if not cv2.imwrite(str(path.with_name(path.stem+'_raw'+path.suffix)),image):raise RuntimeError('Cannot save raw overview')
        image=cv2.rotate(image,{90:cv2.ROTATE_90_CLOCKWISE,180:cv2.ROTATE_180,270:cv2.ROTATE_90_COUNTERCLOCKWISE}[args.rotation_deg])
    if not cv2.imwrite(str(path),image):raise RuntimeError('Cannot save overview')
    path.with_suffix('.json').write_text(json.dumps(dict(device=args.device,captured_unix=time.time(),rotation_deg=args.rotation_deg,width=int(image.shape[1]),height=int(image.shape[0]),metric_extrinsics_verified=False,robot_command_sent=False),indent=2))
    print(path)
finally:camera.release()
