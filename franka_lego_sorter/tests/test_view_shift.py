import json,tempfile,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from validate_view_shift import review

class ViewShiftBoundsTests(unittest.TestCase):
    def reject(self,intent,message):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'plan.json';p.write_text(json.dumps({'commissioning_view_shift':intent}))
            with self.assertRaisesRegex(ValueError,message):review(p)
    def test_unsupported_axis(self):
        self.reject({'camera_axis':'negative_y'},'positive cameraY')
    def test_oversized_shift(self):
        self.reject({'camera_axis':'positive_y_projected_xy','distance_m':.061},'atmost60mm')
    def test_zero_shift(self):
        self.reject({'camera_axis':'positive_y_projected_xy','distance_m':0.},'positive')
