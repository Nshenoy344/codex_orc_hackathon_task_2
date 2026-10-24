from copy import deepcopy
import json
import unittest
import cv2
import numpy as np
from lego_sorter.camera import Frame
from lego_sorter.perception import ColorModel
from lego_sorter.source_slots import ObservedSourceSlots,SourceSlotsError,image_hash,inspect_source_slots


class ObservedSourceSlotTests(unittest.TestCase):
    def scene(self):
        color=np.full((180,300,3),160,np.uint8)
        slots=[]
        for ridge in range(6):
            for position in range(5):
                # Deliberate unequal staggering/spacing, retained in image units.
                x=15+45*ridge+position%2
                y=15+30*position+(ridge%3)
                polygon=[[x,y],[x+23,y-1],[x+24,y+22],[x,y+23]]
                cv2.fillConvexPoly(color,np.array(polygon,np.int32),(18,13,30))
                slots.append(dict(slot_id=f'r{ridge}b{position}',ridge_index=ridge,ridge_position=position,
                    observed_face_polygon_px=polygon,observed_center_px=np.array(polygon).mean(axis=0).tolist(),
                    polygon_role='reference_face_association_only'))
        frame=Frame(color,np.full((180,300),.3,np.float32),dict(fx=500.,fy=500.,ppx=150.,ppy=90.),
                    0.,dict(serial='test',width=300,height=180,fps=30))
        document=dict(format='manually_observed_source_slots_v1',source_snapshot='synthetic',
            source_snapshot_sha256='synthetic_hash',source_color_sha256=image_hash(color),
            source_image_shape=list(color.shape),camera_metadata=frame.metadata,slots=slots)
        return frame,document

    def model(self,metadata):
        model=ColorModel()
        for label,color in [('dark',(18,13,30)),('tan',(100,160,190))]:
            model.teach(label,np.full((30,30,3),color,np.uint8),[2,2,26,26])
        model.record_camera(metadata)
        return model

    def test_all30_dark_faces_classified_without_lightness_segmentation(self):
        frame,document=self.scene()
        self.assertLess(cv2.cvtColor(np.uint8([[[18,13,30]]]),cv2.COLOR_BGR2LAB)[0,0,0],30)
        result=inspect_source_slots(frame,ObservedSourceSlots(document),self.model(frame.metadata))
        self.assertEqual(result['slot_count'],30)
        self.assertEqual(result['classifications'],{'dark':30})
        self.assertFalse(result['execution_eligible'])
        self.assertTrue(all(not slot['geometry']['eligible_for_contact'] for slot in result['slots']))

    def test_unknown_cell_is_reported_and_never_auto_rejected(self):
        frame,document=self.scene()
        polygon=document['slots'][7]['observed_face_polygon_px']
        cv2.fillConvexPoly(frame.color_bgr,np.array(polygon,np.int32),(0,255,0))
        document['source_color_sha256']=image_hash(frame.color_bgr)
        result=inspect_source_slots(frame,ObservedSourceSlots(document),self.model(frame.metadata))
        self.assertEqual(result['classifications'].get('UNKNOWN'),1)
        slot=result['slots'][7]
        self.assertIsNone(slot['color_id'])
        self.assertFalse(slot['execution_eligible'])
        self.assertNotIn('destination',slot)

    def test_changed_scene_or_profile_requires_new_measured_registration(self):
        frame,document=self.scene();registry=ObservedSourceSlots(document)
        frame.color_bgr[1,1]=[1,2,3]
        with self.assertRaisesRegex(SourceSlotsError,'Reference image changed'):
            inspect_source_slots(frame,registry,self.model(frame.metadata))
        frame,document=self.scene();registry=ObservedSourceSlots(document)
        frame.metadata['serial']='different_camera'
        with self.assertRaisesRegex(SourceSlotsError,'camera/profile differs'):
            registry.validate_frame(frame)

    def test_no_grid_fit_changes_observed_stagger_or_centers(self):
        frame,document=self.scene()
        registry=ObservedSourceSlots(document)
        result=inspect_source_slots(frame,registry,self.model(frame.metadata))
        for observed,classified in zip(document['slots'],result['slots']):
            self.assertEqual(observed['observed_face_polygon_px'],classified['observed_face_polygon_px'])
            self.assertEqual(observed['observed_center_px'],classified['observed_center_px'])

    def test_missing_or_duplicate_cell_does_not_create_invented_slot(self):
        _,document=self.scene()
        for mutation in ('missing','duplicate'):
            corrupt=deepcopy(document)
            if mutation=='missing':corrupt['slots'].pop()
            else:corrupt['slots'][-1]=deepcopy(corrupt['slots'][0])
            with self.assertRaisesRegex(SourceSlotsError,'Exactly30 distinct'):
                ObservedSourceSlots(corrupt)

    def test_confirmed_training_face_consistency_is_not_accuracy_claim(self):
        frame,document=self.scene()
        document['slots'][0]['confirmed_identity']=dict(code='M1',physical_color_id='dark',independent_test_label=False)
        result=inspect_source_slots(frame,ObservedSourceSlots(document),self.model(frame.metadata))
        self.assertTrue(result['slots'][0]['teaching_object_consistent'])
        self.assertIsNone(result['slots'][1]['teaching_object_consistent'])
        self.assertIn('not an independent accuracy evaluation',result['limitations'])


if __name__=='__main__':unittest.main()
