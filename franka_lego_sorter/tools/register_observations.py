"""Register measured stationary RGBD observations; does not move robot."""
import argparse,json
from pathlib import Path
import numpy as np
from lego_sorter.camera import Frame
from lego_sorter.scene_registration import register_rgbd

def load(path):
    data=np.load(path,allow_pickle=False)
    frame=Frame(data['color_bgr'],data['depth_m'],json.loads(str(data['intrinsics_json'])),
                float(data['captured_monotonic']),json.loads(str(data['metadata_json'])))
    return frame,data['T_base_tcp'] if 'T_base_tcp' in data else None

p=argparse.ArgumentParser();p.add_argument('reference');p.add_argument('current');p.add_argument('--output',required=True)
a=p.parse_args()
f0,g0=load(a.reference);f1,g1=load(a.current)
r=register_rgbd(f0,f1)
data=r.to_dict();data['reference']=a.reference;data['current']=a.current
if g0 is not None and g1 is not None:
    data['T_base_tcp_reference']=g0.tolist();data['T_base_tcp_current']=g1.tolist()
Path(a.output).write_text(json.dumps(data,indent=2))
print(json.dumps({k:v for k,v in data.items() if k not in ('matches_reference_px','matches_current_px','inlier_mask')},indent=2))
