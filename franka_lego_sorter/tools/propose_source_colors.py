#!/usr/bin/env python3
"""Offline provisional shade labels for the recorded source_720_view snapshot.

These manually inspected interior patches are proposals, not confirmed physical
IDs. Output deliberately is NOT a production ColorModel format. No robot/camera
calls and no writes to production colors.json occur.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lego_sorter.perception import lab_feature,roi_mask

# Pixel coordinates are specific to the inspected immutable1280x720 snapshot.
PROPOSALS={
 '300121':dict(name='red',route='good',prefix='R',centers=[(1064,100),(961,225),(861,350)]),
 '6117418':dict(name='maroon / dark red',route='good',prefix='M',centers=[(1064,208),(1168,318),(1064,419)]),
 '6487946':dict(name='orange-brown',route='good',prefix='O',centers=[(861,171),(1168,218),(767,285)]),
 '6135191':dict(name='tan',route='scrap',prefix='T',centers=[(1167,115),(962,515),(674,340)]),
 '4211201':dict(name='dark brown',route='scrap',prefix='B',centers=[(961,125),(862,520),(755,145)]),
}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',default=str(ROOT/'data/commissioning/source_720_view.npz'))
    p.add_argument('--output',default=str(ROOT/'data/commissioning/source_720_color_proposals'))
    a=p.parse_args();snapshot=Path(a.snapshot)
    with np.load(snapshot,allow_pickle=False) as data:
        image=data['color_bgr'].copy();metadata=json.loads(str(data['metadata_json']))
        capture=float(data['captured_monotonic'])
    if image.shape!=(720,1280,3):raise ValueError('Patch coordinates require the inspected1280x720 source view')
    canvas=np.full((800,1780,3),245,np.uint8);canvas[:720,:1280]=image
    report=dict(format='provisional_physical_color_proposals_v1',proposal_revision=2,operator_confirmed=False,
        revision_notes=[dict(removed_code='B3',removed_center_px=[1165,520],
            reason='Human visual review identified the former patch as tan under shadow, not a reliable dark-brown exemplar.',
            replacement_center_px=[755,145],replacement_note='Second ridge top brown brick; patch moved slightly below the suggested130px row to avoid its darkest stud shadow. Physical ID remains provisional.')],
        eligible_for_production=False,production_model_written=False,
        warning='These IDs are visual proposals only. Confirm each physical shade against its label before creating a production model. Camera lighting remains auto; this is not an accuracy evaluation.',
        source_snapshot=str(snapshot.resolve()),source_snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        source_image_shape=list(image.shape),source_roi_px=[630,25,590,565],
        camera_metadata=metadata,captured_monotonic=capture,
        feature_convention='OpenCV8-bit Lab; robust median after15-85% lightness trimming',
        identity_provenance=dict(specification='Screenshot from2026-10-24 13-30-06.png',
            assistance='Visual comparison with labeled seed photographs; no color values transferred across cameras',
            seed_files=['LEGO_L0_300121.jpg','LEGO_L0_6117418.jpg','LEGO_L0_6487946.jpg','LEGOS_L0_6135191.jpg'],
            dark_brown_note='No individually labeled4211201 seed photograph was found; this proposal especially requires operator confirmation'),
        proposals=[])
    cv2.putText(canvas,'PROVISIONAL - NOT CONFIRMED',(1295,28),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,180),2)
    cv2.putText(canvas,'Confirm physical IDs before production',(1295,52),cv2.FONT_HERSHEY_SIMPLEX,.50,(20,20,20),1)
    for row,(label,proposal) in enumerate(PROPOSALS.items()):
        entries=[];top=85+row*132
        cv2.putText(canvas,f"{label}  {proposal['name']}",(1295,top),cv2.FONT_HERSHEY_SIMPLEX,.62,(20,20,20),2)
        cv2.putText(canvas,f"{proposal['route'].upper()} - identity provisional",(1295,top+22),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,0,160),1)
        for j,(x,y) in enumerate(proposal['centers']):
            code=f"{proposal['prefix']}{j+1}";roi=[x-5,y-5,11,11]
            feature=lab_feature(image,roi_mask(image.shape,roi))
            patch=image[y-5:y+6,x-5:x+6];median=np.median(patch.reshape(-1,3),axis=0)
            entries.append(dict(code=code,center_px=[x,y],roi_px=roi,robust_lab=feature.tolist(),
                                median_bgr=median.tolist(),operator_confirmed=False,
                                selection_note='Interior top-face patch; lightness tails trimmed to reduce stud glare/shadow'))
            cv2.rectangle(canvas,(x-6,y-6),(x+6,y+6),(0,255,255),2)
            cv2.putText(canvas,code,(x+8,y-5),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,0,0),3)
            cv2.putText(canvas,code,(x+8,y-5),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1)
            zoom=cv2.resize(patch,(52,52),interpolation=cv2.INTER_NEAREST)
            px=1295+j*151;canvas[top+33:top+85,px:px+52]=zoom
            cv2.putText(canvas,code,(px+58,top+50),cv2.FONT_HERSHEY_SIMPLEX,.45,(20,20,20),1)
            cv2.putText(canvas,f'{x},{y}',(px+58,top+69),cv2.FONT_HERSHEY_SIMPLEX,.36,(20,20,20),1)
            cv2.putText(canvas,','.join(f'{v:.0f}'for v in feature),(px,top+101),cv2.FONT_HERSHEY_SIMPLEX,.37,(20,20,20),1)
        report['proposals'].append(dict(proposed_color_id=label,display_name=proposal['name'],
            proposed_route=proposal['route'],operator_confirmed=False,patches=entries))
    # Show nearest inter-class prototype separation, without claiming performance.
    pairwise=[]
    for i,a1 in enumerate(report['proposals']):
        for b1 in report['proposals'][:i]:
            distance=min(np.linalg.norm(np.array(pa['robust_lab'])-np.array(pb['robust_lab']))
                         for pa in a1['patches']for pb in b1['patches'])
            pairwise.append(dict(proposed_labels=[a1['proposed_color_id'],b1['proposed_color_id']],
                                 minimum_selected_patch_lab_distance=float(distance)))
    report['selected_patch_separations_not_validation']=pairwise
    cv2.putText(canvas,'Source coordinates retained. R=red M=maroon O=orange-brown T=tan B=dark brown.',(20,751),cv2.FONT_HERSHEY_SIMPLEX,.65,(10,10,10),1)
    cv2.putText(canvas,'No physical ID confirmation, production classifier, or sorting success is implied.',(20,783),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,180),1)
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    if not cv2.imwrite(str(output.with_suffix('.png')),canvas):raise RuntimeError('Could not save annotation')
    print(json.dumps(dict(proposal_json=str(output.with_suffix('.json')),annotation=str(output.with_suffix('.png')),
                         labels=[v['proposed_color_id']for v in report['proposals']],confirmed=False,production_model_written=False),indent=2))


if __name__=='__main__':main()
