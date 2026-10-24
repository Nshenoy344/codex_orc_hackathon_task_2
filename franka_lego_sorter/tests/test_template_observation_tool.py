import importlib.util
from pathlib import Path
import unittest
import cv2
import numpy as np
from lego_sorter.camera import Frame
spec=importlib.util.spec_from_file_location('template_observation_tool',Path(__file__).parents[1]/'tools/inspect_template_observation.py');tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)

class ReadonlyTemplateObservationTests(unittest.TestCase):
    def frame(self,count=12):
        hsv=np.zeros((450,400,3),np.uint8)
        for n in range(count):
            col,row=divmod(n,3);hsv[60+row*100:124+row*100,50+col*75:82+col*75]=[15,220,200]
        return Frame(cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR),np.full((450,400),.5,np.float32),dict(fx=1000.,fy=1000.,ppx=200.,ppy=225.),0.,{})
    def test_twelve_individual_faces_group_without_authorizing_execution(self):
        report,_=tool.inspect(self.frame(),[0,0,400,450],[[10,350],[390,350],[390,440],[10,440]])
        self.assertIsNone(report['failure']);self.assertEqual(report['measured_face_count'],12)
        self.assertEqual(len(report['registration']['slots']),12)
        self.assertFalse(report['registration']['placement_ready']);self.assertFalse(report['execution_eligible'])
        self.assertAlmostEqual(report['registration']['support_height_above_table_m'],0.)
    def test_missing_face_does_not_generate_twelfth(self):
        report,_=tool.inspect(self.frame(11),[0,0,400,450]);self.assertIsNone(report['registration']);self.assertEqual(report['measured_face_count'],11)
    def test_clipped_roi_does_not_claim_complete_face(self):
        report,_=tool.inspect(self.frame(),[55,0,345,450]);self.assertIsNone(report['registration']);self.assertTrue(any('boundary' in reason for p in report['proposals'] for reason in p['rejection_reasons']))
if __name__=='__main__':unittest.main()
