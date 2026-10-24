#!/usr/bin/env python3
"""Build the operator-confirmed revision2 teaching model and review recorded faces.

Offline only. Classification of teaching-derived source faces is an operational
review, not an independent accuracy evaluation or a robot-execution command.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import yaml
cv2.setNumThreads(1)
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lego_sorter.camera import Frame
from lego_sorter.perception import ColorModel,lab_feature,roi_mask
from lego_sorter.pallet import inspect_pallet


# Interior-face polygons are traced from the visible brick containing each
# operator-confirmed patch, not assigned by nearest color or an invented grid.
FACE_POLYGONS={
 '300121':[[[1030,62],[1094,58],[1097,164],[1036,170]],[[931,175],[988,171],[991,274],[935,279]],[[834,280],[889,277],[893,375],[838,380]]],
 '6117418':[[[1037,167],[1098,163],[1102,263],[1041,270]],[[1141,265],[1197,262],[1196,362],[1142,366]],[[1044,366],[1107,362],[1108,466],[1046,470]]],
 '6487946':[[[831,79],[883,74],[887,177],[835,182]],[[1140,162],[1199,158],[1197,264],[1140,269]],[[735,195],[790,191],[792,287],[739,292]]],
 '6135191':[[[1136,56],[1199,53],[1198,161],[1141,166]],[[940,465],[995,461],[995,558],[944,561]],[[649,293],[698,290],[701,389],[653,393]]],
 '4211201':[[[926,70],[984,65],[988,171],[931,176]],[[840,465],[894,461],[895,563],[845,568]],[[729,94],[783,90],[788,191],[735,196]]],
}


def load(path):
    with np.load(path,allow_pickle=False)as d:
        return Frame(d['color_bgr'],d['depth_m'],json.loads(str(d['intrinsics_json'])),float(d['captured_monotonic']),json.loads(str(d['metadata_json'])))


def color_region(frame,box):
    mask=np.zeros(frame.depth_m.shape,np.uint8)
    cv2.fillConvexPoly(mask,np.round(box).astype(np.int32),255)
    # Physical-color review excludes edges, the brightest stud reflections, and
    # deepest visible shadows; the same robustLab routine is used by production.
    mask=cv2.erode(mask,np.ones((7,7),np.uint8))
    if np.count_nonzero(mask)<25:raise ValueError('Too little interior color support')
    # ColorModel.classify/lab_feature performs the sole15-85% lightness trim.
    return mask


def classified_faces(frame,settings,model):
    model.validate_camera(frame.metadata)
    inspected=inspect_pallet(frame,settings,model)
    report=inspected.to_dict()
    for candidate in report['candidates']:
        mask=color_region(frame,np.array(candidate['box_px']))
        c=model.classify(frame.color_bgr,mask)
        score=max(0.,1-c.distance/model.max_distance) if c.label is not None else 0.
        candidate.update(color_id=c.label,color_distance=c.distance,color_margin=c.margin,
            color_confidence_score_not_probability=score,robust_interior_lab=c.feature_lab,
            color_pixel_count=int(np.count_nonzero(mask)))
    report['review_provenance']=dict(camera=frame.metadata,
        evaluation_scope='Recorded source candidate faces, not an independent physical accuracy evaluation',
        geometry_warning='This is every returned geometry proposal, not proof that every true top brick was detected; split/occlusion/graspability reasons remain in each candidate')
    known={}
    for c in report['candidates']:
        label=c['color_id']or'UNKNOWN';known[label]=known.get(label,0)+1
    report['classification_summary']=dict(candidate_count=len(report['candidates']),counts_by_label=known,
        unknown_count=known.get('UNKNOWN',0),max_distance=model.max_distance,min_margin=model.min_margin,
        independent_accuracy_evaluation=False)
    return report


def annotate(frame,report,title):
    h,w=frame.color_bgr.shape[:2]
    canvas=np.full((h+170,w+520,3),245,np.uint8);canvas[:h,:w]=frame.color_bgr
    cv2.putText(canvas,title,(w+10,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(20,20,20),1)
    cv2.putText(canvas,'score is NOT a probability',(w+10,48),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,0,170),1)
    palette={'300121':(40,40,230),'6117418':(140,70,170),'6487946':(10,150,230),
             '6135191':(70,170,200),'4211201':(110,160,130),None:(180,180,180)}
    for index,c in enumerate(report['candidates']):
        label=c['color_id']or'UNKNOWN';color=palette[c['color_id']]
        points=np.round(c['box_px']).astype(np.int32)
        cv2.polylines(canvas,[points],True,color,2)
        x,y=np.round(c['center_px']).astype(int);score=c['color_confidence_score_not_probability']
        for dy,text in [(0,f'{index}:{label}'),(13,f's={score:.2f} m={c["color_margin"]:.1f}')]:
            cv2.putText(canvas,text,(x-28,y+dy),cv2.FONT_HERSHEY_SIMPLEX,.32,(0,0,0),3)
            cv2.putText(canvas,text,(x-28,y+dy),cv2.FONT_HERSHEY_SIMPLEX,.32,(255,255,255),1)
        row=index%30;column=index//30
        cv2.putText(canvas,f'{index:02d} {label} s={score:.2f} gap={c["color_margin"]:.1f}',
                    (w+10+column*260,78+row*20),cv2.FONT_HERSHEY_SIMPLEX,.42,(20,20,20),1)
    summary=report['classification_summary']
    cv2.putText(canvas,f"Geometry proposals:{summary['candidate_count']}  UNKNOWN:{summary['unknown_count']}  distance<=18 / margin>=5 unchanged",
                (15,h+35),cv2.FONT_HERSHEY_SIMPLEX,.58,(20,20,20),1)
    cv2.putText(canvas,'Teaching/source review only - not an independent accuracy or picking demonstration.',
                (15,h+70),cv2.FONT_HERSHEY_SIMPLEX,.58,(0,0,170),1)
    cv2.putText(canvas,'Uncertain faces stay UNKNOWN; candidate geometry and grasp-access rejection reasons are retained.',
                (15,h+105),cv2.FONT_HERSHEY_SIMPLEX,.55,(20,20,20),1)
    return canvas


def isolated_color_review(isolated,iso,model):
    try:
        iso_report=classified_faces(isolated,iso,model)
        if not iso_report['candidates']:
            raise ValueError('No geometric candidate found; isolated visible-face color review is separate')
    except ValueError as exc:
        # The isolated visible polygon is a color-only diagnostic, not a new
        # geometric grasp proposal or evidence of its supporting depth reference.
        polygon=np.array([[858,601],[888,626],[838,681],[808,655]],float)
        mask=color_region(isolated,polygon);classified=model.classify(isolated.color_bgr,mask)
        label=classified.label;score=max(0.,1-classified.distance/model.max_distance) if label is not None else 0.
        iso_report=dict(candidates=[dict(center_px=polygon.mean(axis=0).tolist(),box_px=polygon.tolist(),
            color_id=label,color_distance=classified.distance,color_margin=classified.margin,
            color_confidence_score_not_probability=score,robust_interior_lab=classified.feature_lab,
            graspable=False,rejection_reasons=['manual_color_review_only_no_registered_grasp_geometry'],
            geometry_source='Visible isolated-brick polygon traced for color review only')],
            inspection_error=str(exc),classification_summary=dict(candidate_count=1,
                counts_by_label={label or'UNKNOWN':1},unknown_count=int(label is None),max_distance=18.,min_margin=5.,
                independent_accuracy_evaluation=False),review_provenance=dict(evaluation_scope='One isolated color-only diagnostic; not automated grasp detection or independent accuracy validation'))
    return iso_report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-model',default=str(ROOT/'data/colors.json'))
    a=p.parse_args();source=ROOT/'data/commissioning/source_720_view.npz'
    proposal_path=ROOT/'data/commissioning/source_720_color_proposals.json'
    proposal=json.loads(proposal_path.read_text())
    if proposal.get('proposal_revision')!=2:raise ValueError('Only reviewed revision2 is confirmed')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if digest!=proposal['source_snapshot_sha256']:raise ValueError('Teaching source changed after identity confirmation')
    frame=load(source);samples={};regions=[]
    for shade in proposal['proposals']:
        label=shade['proposed_color_id'];samples[label]=[]
        for patch,polygon in zip(shade['patches'],FACE_POLYGONS[label]):
            if cv2.pointPolygonTest(np.array(polygon,np.float32),tuple(map(float,patch['center_px'])),False)<0:
                raise ValueError('A traced face does not contain its confirmed identity anchor')
            feature=lab_feature(frame.color_bgr,color_region(frame,np.array(polygon)))
            samples[label].append(feature.tolist())
            regions.append(dict(code=patch['code'],confirmed_physical_color_id=label,
                confirmed_anchor_roi_px=patch['roi_px'],confirmed_anchor_center_px=patch['center_px'],
                face_polygon_px=polygon,robust_face_lab=feature.tolist(),
                association='Visible same-brick object continuity from the operator-confirmed patch; manually traced face, no color-nearest identity assignment',
                feature_representation='eroded whole-face interior with the same single robustLab trim as production'))
    region_report=dict(format='confirmed_color_identity_face_regions_v1',source_snapshot=str(source),
        source_snapshot_sha256=digest,camera_metadata=frame.metadata,regions=regions,
        independent_accuracy_evaluation=False)
    (ROOT/'data/commissioning/confirmed_face_teaching_regions.json').write_text(json.dumps(region_report,indent=2)+'\n')
    metadata=dict(camera=frame.metadata,physical_color_identities_confirmed=True,
        confirmation_recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        confirmation_source='Operator confirmation relayed by orchestrator: all revision2 physical shade identities correct',
        teaching_proposal_revision=2,teaching_proposal_sha256=hashlib.sha256(proposal_path.read_bytes()).hexdigest(),
        source_snapshot=str(source),source_snapshot_sha256=digest,
        physical_teaching_rois={s['proposed_color_id']:[p['roi_px']for p in s['patches']]for s in proposal['proposals']},
        confirmation_scope='Physical shade identities and anchor objects; not classifier accuracy, grasp success, or camera-scale verification',
        feature_representation='confirmed_same_brick_whole_face_interior',
        face_regions=regions,
        representation_change_reason='Use the same whole-face feature representation in teaching and production; tinybodypatch shadow intensity is not representative of the entire face',
        independent_accuracy_evaluation=False)
    model=ColorModel(samples,max_distance=18.,min_margin=5.,metadata=metadata)
    model.validate_labels(['300121','6117418','6487946','6135191','4211201'])
    output=Path(a.output_model);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        backup=output.with_name(output.stem+'.before_confirmed_revision2'+output.suffix)
        if backup.exists():raise ValueError('Backup exists; refuse to overwrite another production model')
        backup.write_bytes(output.read_bytes())
    payload=dict(format='opencv_lab_v1',samples=samples,metadata=metadata)
    temp=output.with_suffix(output.suffix+'.tmp');temp.write_text(json.dumps(payload,indent=2)+'\n');temp.replace(output)
    config=yaml.safe_load((ROOT/'config/demo.yaml').read_text())
    settings=dict(config['perception']);settings.update(config.get('pallet',{}))
    settings['roi']=[630,25,590,565]
    settings.update(grasp_depth_below_surface_m=config['execution']['finger_insertion_below_surface_m'],
                    jaw_opening_margin_m=config['execution']['gripper_opening_margin_m'])
    # Include independent white table in table-depth estimate; the tight source
    # ROI is mostly raised bricks and is not itself a table-depth measurement.
    whole=frame.depth_m;table=whole[350:650,500:600];valid=table[np.isfinite(table)&(table>0)]
    if len(valid)>100:settings['table_depth_m']=float(np.median(valid))
    report=classified_faces(frame,settings,model)
    report['confirmed_color_model']=str(output);report['color_model_sha256']=hashlib.sha256(output.read_bytes()).hexdigest()
    base=ROOT/'data/commissioning/source_720_classified'
    base.with_suffix('.json').write_text(json.dumps(report,indent=2))
    cv2.imwrite(str(base.with_suffix('.png')),annotate(frame,report,'Confirmed source-face review'))
    isolated=load(ROOT/'data/commissioning/isolated_720_view.npz')
    iso=dict(settings,roi=[790,580,115,115],table_depth_m=None)
    iso_report=isolated_color_review(isolated,iso,model)
    iso_base=ROOT/'data/commissioning/isolated_720_classified'
    iso_base.with_suffix('.json').write_text(json.dumps(iso_report,indent=2))
    cv2.imwrite(str(iso_base.with_suffix('.png')),annotate(isolated,iso_report,'Isolated test brick review'))
    print(json.dumps(dict(production_model=str(output),source=report['classification_summary'],isolated=iso_report['classification_summary']),indent=2))


if __name__=='__main__':main()
