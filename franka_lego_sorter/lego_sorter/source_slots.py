"""Read-only association/classification of30 actually observed source-face polygons.

This registry describes one immutable observed top layer, not a generated grid or
an execution source map. Reuse on a changed view/layer requires explicit measured
registration, which this first offline tool deliberately does not invent.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from .camera import Frame
from .pallet import fit_surface_plane


class SourceSlotsError(ValueError):
    pass


def image_hash(image):
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


class ObservedSourceSlots:
    def __init__(self, document):
        self.document=deepcopy(document)
        d=self.document
        if d.get('format')!='manually_observed_source_slots_v1':
            raise SourceSlotsError('Unknown observed-source registry format')
        slots=d.get('slots',[])
        expected={(r,b) for r in range(6) for b in range(5)}
        actual={(slot.get('ridge_index'),slot.get('ridge_position')) for slot in slots}
        if len(slots)!=30 or actual!=expected or len({s.get('slot_id') for s in slots})!=30:
            raise SourceSlotsError('Exactly30 distinct observed6ridge×5face slot associations are required')
        shape=d.get('source_image_shape')
        if not isinstance(shape,list) or len(shape)!=3 or min(shape)<=0 or shape[2]!=3:
            raise SourceSlotsError('Registry needs the actual reference image dimensions')
        height,width=shape[:2]
        for slot in slots:
            expected_id=f"r{slot['ridge_index']}b{slot['ridge_position']}"
            if slot['slot_id']!=expected_id:
                raise SourceSlotsError('Slot IDs must match their explicit observed ridge/position')
            polygon=np.asarray(slot.get('observed_face_polygon_px'),dtype=np.float32)
            if (polygon.ndim!=2 or polygon.shape[1]!=2 or len(polygon)<4 or
                    not np.isfinite(polygon).all() or not cv2.isContourConvex(polygon) or
                    abs(cv2.contourArea(polygon))<25 or np.any(polygon[:,0]<0) or
                    np.any(polygon[:,0]>=width) or np.any(polygon[:,1]<0) or np.any(polygon[:,1]>=height)):
                raise SourceSlotsError(f'{expected_id}: measured face polygon is invalid or clipped')
            if slot.get('polygon_role')!='reference_face_association_only':
                raise SourceSlotsError('Source-face polygons must not silently claim executable grasp boundaries')
        self.slots=sorted(slots,key=lambda s:(s['ridge_index'],s['ridge_position']))

    @classmethod
    def load(cls,path):
        return cls(json.loads(Path(path).read_text()))

    def to_dict(self):
        return deepcopy(self.document)

    def validate_frame(self,frame):
        d=self.document
        if list(frame.color_bgr.shape)!=d['source_image_shape'] or frame.depth_m.shape!=frame.color_bgr.shape[:2]:
            raise SourceSlotsError('Source frame/profile changed; observed polygons require measured registration')
        if image_hash(frame.color_bgr)!=d.get('source_color_sha256'):
            raise SourceSlotsError('Reference image changed; do not reuse source slots without measured image/layer registration')
        expected=d.get('camera_metadata',{})
        for key in ('serial','width','height'):
            if key in expected and frame.metadata.get(key)!=expected[key]:
                raise SourceSlotsError('Source camera/profile differs from the observed registry')


def load_reference_frame(registry):
    path=Path(registry.document['source_snapshot'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=registry.document.get('source_snapshot_sha256'):
        raise SourceSlotsError('Reference snapshot hash changed')
    with np.load(path,allow_pickle=False) as data:
        frame=Frame(data['color_bgr'].copy(),data['depth_m'].copy(),json.loads(str(data['intrinsics_json'])),
                    float(data['captured_monotonic']),json.loads(str(data['metadata_json'])))
    registry.validate_frame(frame)
    return frame


def inspect_source_slots(frame,registry,model):
    registry.validate_frame(frame)
    model.validate_camera(frame.metadata)
    results=[]
    for slot in registry.slots:
        polygon=np.asarray(slot['observed_face_polygon_px'],np.float32)
        mask=np.zeros(frame.depth_m.shape,np.uint8)
        cv2.fillConvexPoly(mask,np.round(polygon).astype(np.int32),255)
        # Same seven-pixel whole-face erosion used by confirmed physical exemplars.
        mask=cv2.erode(mask,np.ones((7,7),np.uint8))
        area=int(np.count_nonzero(mask))
        if area<20:raise SourceSlotsError(f"{slot['slot_id']}: too little actual face interior")
        shade=model.classify(frame.color_bgr,mask)
        valid=(mask>0)&np.isfinite(frame.depth_m)&(frame.depth_m>0)
        values=frame.depth_m[valid]
        geometry=dict(valid_depth_fraction=float(len(values)/area),surface_reference='unverified_body_top',
                      eligible_for_contact=False)
        if len(values):
            median=float(np.median(values))
            geometry.update(visible_depth_median_m=median,
                visible_depth_mad_m=float(np.median(np.abs(values-median))))
            try:
                center,normal,rms=fit_surface_plane(frame,valid)
                pixel=np.array(slot['observed_center_px'],float)
                ray=frame.deproject(float(pixel[0]),float(pixel[1]),1.)
                denominator=float(normal@ray)
                if abs(denominator)<.1:raise ValueError('Observed plane is too oblique')
                point=ray*float((normal@center)/denominator)
                geometry.update(visible_face_point_camera_m=point.tolist(),visible_face_plane_normal_camera=normal.tolist(),
                                visible_face_plane_rms_m=float(rms))
            except ValueError as error:
                geometry['depth_reason']=str(error)
        reference_identity=slot.get('confirmed_identity')
        results.append(dict(slot_id=slot['slot_id'],ridge_index=slot['ridge_index'],
            ridge_position=slot['ridge_position'],observed_center_px=slot['observed_center_px'],
            observed_face_polygon_px=slot['observed_face_polygon_px'],color_id=shade.label,
            color_distance=float(shade.distance),color_margin=float(shade.margin),feature_lab=shade.feature_lab,
            geometry=geometry,execution_eligible=False,
            teaching_object_identity=reference_identity,
            teaching_object_consistent=None if reference_identity is None else shade.label==reference_identity['physical_color_id']))
    counts={}
    for result in results:
        key=result['color_id'] or 'UNKNOWN'
        counts[key]=counts.get(key,0)+1
    return dict(format='readonly_source_slot_inspection_v1',reference_only=True,execution_eligible=False,
                source_snapshot=registry.document['source_snapshot'],
                source_snapshot_sha256=registry.document['source_snapshot_sha256'],
                slot_count=30,classifications=counts,slots=results,
                limitations='No live registration, body-top proof, occupancy commit or grasp authorization. Teaching-object consistency is not an independent accuracy evaluation.')


def annotate_source_slots(frame,inspection):
    image=frame.color_bgr.copy()
    for result in inspection['slots']:
        label=result['color_id'] or 'UNKNOWN'
        color=(0,200,0) if result['color_id'] else (0,0,255)
        polygon=np.round(result['observed_face_polygon_px']).astype(np.int32)
        cv2.polylines(image,[polygon],True,color,1)
        center=tuple(np.round(result['observed_center_px']).astype(int))
        cv2.circle(image,center,3,color,-1)
        text=f"{result['slot_id']} {label}"
        cv2.putText(image,text,(center[0]-24,center[1]),cv2.FONT_HERSHEY_SIMPLEX,.30,(0,0,0),3)
        cv2.putText(image,text,(center[0]-24,center[1]),cv2.FONT_HERSHEY_SIMPLEX,.30,(255,255,255),1)
    return image


def main():
    import argparse
    from .perception import ColorModel
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('registry')
    parser.add_argument('--model',required=True)
    parser.add_argument('--output',default='data/commissioning/source_slot_inspection')
    args=parser.parse_args()
    registry=ObservedSourceSlots.load(args.registry)
    frame=load_reference_frame(registry)
    model=ColorModel.load(args.model)
    inspection=inspect_source_slots(frame,registry,model)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.with_suffix('.json').write_text(json.dumps(inspection,indent=2))
    if not cv2.imwrite(str(output.with_suffix('.png')),annotate_source_slots(frame,inspection)):
        raise RuntimeError('Could not save source association overlay')
    print(json.dumps({key:value for key,value in inspection.items() if key!='slots'},indent=2))


if __name__=='__main__':main()
