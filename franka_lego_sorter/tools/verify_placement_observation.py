#!/usr/bin/env python3
"""Offline comparison only; never moves robot or commits occupancy."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np,yaml
from lego_sorter.camera import Frame
from lego_sorter.perception import ColorModel
from lego_sorter.placement_verification import verify_placement
p=argparse.ArgumentParser(description=__doc__);p.add_argument('before',type=Path);p.add_argument('after',type=Path);p.add_argument('--registration',type=Path,required=True);p.add_argument('--slot-id',required=True);p.add_argument('--handeye',type=Path,required=True);p.add_argument('--colors',type=Path,default=Path('data/colors.json'));p.add_argument('--held-color',default='6117418');p.add_argument('--output',type=Path,required=True);a=p.parse_args();cv2.setNumThreads(1)
def load(path):
 with np.load(path,allow_pickle=False) as d:return Frame(d['color_bgr'],d['depth_m'],json.loads(str(d['intrinsics_json'])),float(d['captured_monotonic']),json.loads(str(d['metadata_json']))),np.array(d['T_base_tcp'],float)
before,Tbefore=load(a.before);after,Tafter=load(a.after);candidate=yaml.safe_load(a.handeye.read_text())
if candidate.get('diagnostics',{}).get('accepted') is not True:raise ValueError('Numerically rejected calibration candidate')
T=np.array(candidate['T_tcp_color'],float);registry=json.loads(a.registration.read_text());slots=registry['observed_slots'];slot=next(s for s in slots if s['slot_id']==a.slot_id)
import hashlib
if registry['snapshot_sha256']!=hashlib.sha256(a.before.read_bytes()).hexdigest():raise ValueError('Before frame differs from registered actual observation')
try:report,mask=verify_placement(before,after,Tbefore@T,Tafter@T,slot,ColorModel.load(a.colors),a.held_color)
except ValueError as error:report=dict(status='UNKNOWN',reason=str(error),execution_eligible=False,production_counts_committed=False);mask=None
report.update(before_snapshot=str(a.before),after_snapshot=str(a.after),slot_id=a.slot_id,commissioning_candidate_used=True)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.with_suffix('.json').write_text(json.dumps(report,indent=2));image=after.color_bgr.copy()
if mask is not None:cv2.polylines(image,[np.round(report['after_observed_footprint_px']).astype(int)],True,(0,255,255),2)
cv2.imwrite(str(a.output.with_suffix('.png')),image);print(json.dumps(report,indent=2))
