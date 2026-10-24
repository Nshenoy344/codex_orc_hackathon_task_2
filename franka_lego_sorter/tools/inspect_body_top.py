#!/usr/bin/env python3
"""Offline RGBD body/stud reference inspection; never opens a camera or robot.

Example:
  PYTHONNOUSERSITE=1 /usr/bin/python3 tools/inspect_body_top.py \
    data/commissioning/source_1280.npz --config config/demo.yaml
"""
import argparse
import json
from pathlib import Path
import sys
import cv2
cv2.setNumThreads(1)
import numpy as np
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lego_sorter.camera import Frame
from lego_sorter.pallet import inspect_pallet, annotate
from lego_sorter.perception import ColorModel


def load_frame(path):
    with np.load(path,allow_pickle=False) as data:
        required=('color_bgr','depth_m','intrinsics_json','captured_monotonic')
        if any(key not in data for key in required):
            raise ValueError('Snapshot needs aligned color/depth and actual intrinsics; not a color PNG')
        metadata=json.loads(str(data['metadata_json'])) if 'metadata_json' in data else {}
        return Frame(data['color_bgr'].copy(),data['depth_m'].copy(),
                     json.loads(str(data['intrinsics_json'])),float(data['captured_monotonic']),metadata)


def profile_summary(frame):
    valid=frame.depth_m[np.isfinite(frame.depth_m)&(frame.depth_m>0)]
    levels=np.unique(np.round(valid,6))
    steps=np.diff(levels)
    return dict(aligned_image_width=int(frame.color_bgr.shape[1]),
                aligned_image_height=int(frame.color_bgr.shape[0]),
                intrinsics=frame.intrinsics,camera_metadata=frame.metadata,
                valid_depth_fraction=float(len(valid)/frame.depth_m.size),
                median_positive_stored_depth_step_m=float(np.median(steps[steps>0])) if np.any(steps>0) else None,
                depth_step_warning='Stored numeric increments do not establish absolute depth accuracy or physical surface resolution',
                native_depth_profile_recorded=any(key in frame.metadata for key in ('native_depth_width','depth_width','depth_native_width')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshots',nargs='+')
    parser.add_argument('--config',default='config/demo.yaml')
    parser.add_argument('--model',help='Optional actual taught physical shade model; no model means unclassified')
    parser.add_argument('--output-dir',default='data/commissioning/body_top_inspection')
    parser.add_argument('--roi',nargs=4,type=int,metavar=('X','Y','WIDTH','HEIGHT'))
    parser.add_argument('--strip-half-width-mm',type=float)
    parser.add_argument('--max-plane-rms-mm',type=float)
    parser.add_argument('--min-strip-area-mm2',type=float)
    parser.add_argument('--min-stud-area-mm2',type=float)
    parser.add_argument('--min-relief-mm',type=float)
    parser.add_argument('--max-relief-mm',type=float)
    parser.add_argument('--require-one-verified',action='store_true',help='Exit2 if no measured body reference passes; does not authorize a pick')
    args=parser.parse_args()
    config=yaml.safe_load(Path(args.config).read_text())
    settings=dict(config.get('perception',{}));settings.update(config.get('pallet',{}))
    execution=config.get('execution',{})
    if 'finger_insertion_below_surface_m' in execution:
        settings['grasp_depth_below_surface_m']=execution['finger_insertion_below_surface_m']
    if 'gripper_opening_margin_m' in execution:
        settings['jaw_opening_margin_m']=execution['gripper_opening_margin_m']
    settings['require_body_top_reference']=True
    if args.roi:settings['roi']=args.roi
    for argument,key,scale in (
            ('strip_half_width_mm','body_top_strip_half_width_m',.001),
            ('max_plane_rms_mm','body_top_max_plane_rms_m',.001),
            ('min_strip_area_mm2','body_top_min_strip_area_m2',.000001),
            ('min_stud_area_mm2','body_top_min_stud_area_m2',.000001),
            ('min_relief_mm','body_top_min_resolved_stud_relief_m',.001),
            ('max_relief_mm','body_top_max_resolved_stud_relief_m',.001)):
        value=getattr(args,argument)
        if value is not None:settings[key]=value*scale
    model=ColorModel.load(args.model) if args.model else None
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    summaries=[]
    verified=0
    for name in args.snapshots:
        snapshot=Path(name)
        frame=load_frame(snapshot)
        report=dict(snapshot=str(snapshot.resolve()),profile=profile_summary(frame),settings=settings)
        image=frame.color_bgr.copy()
        try:
            inspection=inspect_pallet(frame,settings,model)
            report['inspection']=inspection.to_dict()
            image=annotate(frame,inspection)
            verified+=inspection.diagnostics['body_top_verified_count']
            details=[]
            for index,candidate in enumerate(inspection.candidates):
                evidence=candidate.body_top_diagnostics or {}
                confirmed=candidate.surface_reference=='measured_inter_stud_body_top'
                color=(0,220,0) if confirmed else (0,0,255)
                if 'strip_polygon_px' in evidence:
                    cv2.polylines(image,[np.round(evidence['strip_polygon_px']).astype(np.int32)],True,color,1)
                for point,relief in zip(evidence.get('stud_centers_px',[]),evidence.get('measured_stud_relief_m',[])):
                    cv2.circle(image,tuple(np.round(point).astype(int)),2,color if relief is not None else (100,100,100),1)
                details.append(dict(index=index,center_px=candidate.center_px,
                    body_reference=candidate.surface_reference,
                    relief_mm=[None if value is None else round(value*1000,3) for value in evidence.get('measured_stud_relief_m',[])],
                    body_rms_mm=None if evidence.get('body_plane_rms_m') is None else evidence['body_plane_rms_m']*1000,
                    reason=evidence.get('reason'),graspable=candidate.graspable))
            summary=dict(snapshot=str(snapshot),candidates=len(inspection.candidates),
                measured_body_references=inspection.diagnostics['body_top_verified_count'],
                executable_grasps=inspection.diagnostics['graspable_count'],details=details)
        except ValueError as error:
            report['error']=str(error)
            summary=dict(snapshot=str(snapshot),error=str(error),measured_body_references=0)
        prefix=output/snapshot.stem
        prefix.with_suffix('.json').write_text(json.dumps(report,indent=2))
        if not cv2.imwrite(str(prefix.with_suffix('.png')),image):
            raise RuntimeError('Could not save diagnostic image')
        summaries.append(summary)
    (output/'summary.json').write_text(json.dumps(summaries,indent=2))
    print(json.dumps(summaries,indent=2))
    return 2 if args.require_one_verified and not verified else 0


if __name__=='__main__':
    raise SystemExit(main())
