#!/usr/bin/env python3
"""Read-only orange-face proposals; no inferred slots, height correction or motion."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from lego_sorter.camera import Frame
from lego_sorter.template_registration import _measure_slot_polygon, register_template, TemplateError


def inspect(frame, roi, table_polygon=None, hsv_lower=(3,90,40), hsv_upper=(29,255,255)):
    x,y,w,h=map(int,roi)
    height,width=frame.depth_m.shape
    if min(x,y)<0 or min(w,h)<=0 or x+w>width or y+h>height:
        raise ValueError('ROI must be wholly inside actual image')
    hsv=cv2.cvtColor(frame.color_bgr,cv2.COLOR_BGR2HSV)
    mask=np.zeros((height,width),np.uint8)
    mask[y:y+h,x:x+w]=cv2.inRange(hsv[y:y+h,x:x+w],np.array(hsv_lower),np.array(hsv_upper))
    # Closing repairs single-pixel holes only; no erosion/splitting invents brick boundaries.
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    image=frame.color_bgr.copy();proposals=[];measured=[]
    config=dict(template_type='brick_layer',template_type_verified=False,
                min_slot_depth_fraction=.75,max_slot_plane_rms_m=.003)
    for contour in sorted(contours,key=cv2.contourArea,reverse=True):
        if cv2.contourArea(contour)<40:continue
        rect=cv2.minAreaRect(contour);box=cv2.boxPoints(rect);fill=cv2.contourArea(contour)/max(1,np.prod(rect[1]));reasons=[]
        if np.any(box[:,0]<=x+1) or np.any(box[:,0]>=x+w-2) or np.any(box[:,1]<=y+1) or np.any(box[:,1]>=y+h-2):reasons.append('face touches image/ROI boundary')
        if fill<.70:reasons.append('nonrectangular or merged color region')
        item=None
        try:
            item=_measure_slot_polygon(frame,box,config)
            if abs(item['length_m']-.0318)>.004 or abs(item['width_m']-.0158)>.004:reasons.append('measured footprint not one2x4 brick; no automatic split')
        except (ValueError,TemplateError) as error:reasons.append(str(error))
        proposal=dict(index=len(proposals),polygon_px=box.tolist(),center_px=list(rect[0]),color_region_fill=float(fill),measurement=item,rejection_reasons=reasons)
        proposals.append(proposal)
        if not reasons:measured.append(item)
        color=(0,0,255) if reasons else (0,255,0);cv2.polylines(image,[np.round(box).astype(int)],True,color,2)
        cv2.putText(image,str(proposal['index']),(int(rect[0][0]),int(rect[0][1])),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
    failure=None;registration=None
    if any(p['rejection_reasons'] for p in proposals):failure='Partial/merged/unmeasurable orange regions remain; refine ROI or obtain complete view'
    elif len(measured)!=12:failure=f'Exactly12 independent measured faces required; observed{len(measured)}; no missing slots generated'
    else:
        station=[[x,y],[x+w-1,y],[x+w-1,y+h-1],[x,y+h-1]]
        if table_polygon is not None:config['table_polygon_px']=table_polygon
        try:
            result=register_template(frame,config,station_polygon_px=station,candidates=measured)
            registration=result.to_dict()
            registration['placement_ready']=False
        except (ValueError,TemplateError) as error:failure=str(error)
    return dict(format='readonly_orange_template_observation_v1',roi_px=list(roi),hsv_lower=list(hsv_lower),hsv_upper=list(hsv_upper),
                proposed_face_count=len(proposals),measured_face_count=len(measured),proposals=proposals,registration=registration,failure=failure,
                execution_eligible=False,surface_reference='visible_RGBD_top_only_body_or_stud_reference_unverified',
                limitations='Color is a geometry proposal, not operator-confirmed identity. No face splitting, extrapolated spacing, nominal stud subtraction, bodytop correction, layer count or layer pitch inferred. Table-relative visible-top height is diagnostic only.'),image


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('snapshot',type=Path);parser.add_argument('--roi',nargs=4,type=int,required=True);parser.add_argument('--table-polygon',nargs='+',type=float);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--hsv-lower',nargs=3,type=int,default=[3,90,40]);parser.add_argument('--hsv-upper',nargs=3,type=int,default=[29,255,255]);args=parser.parse_args()
    cv2.setNumThreads(1)
    with np.load(args.snapshot,allow_pickle=False) as data:frame=Frame(data['color_bgr'],data['depth_m'],json.loads(str(data['intrinsics_json'])),float(data['captured_monotonic']),json.loads(str(data['metadata_json'])))
    polygon=None if args.table_polygon is None else np.asarray(args.table_polygon).reshape(-1,2).tolist()
    report,image=inspect(frame,args.roi,polygon,args.hsv_lower,args.hsv_upper);report.update(snapshot=str(args.snapshot.resolve()),snapshot_sha256=hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),camera_metadata=frame.metadata)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.with_suffix('.json').write_text(json.dumps(report,indent=2));cv2.imwrite(str(args.output.with_suffix('.png')),image)
    print(json.dumps({k:report[k] for k in ('proposed_face_count','measured_face_count','failure','execution_eligible')}))
    return 0 if report['registration'] is not None else 2

if __name__=='__main__':raise SystemExit(main())
