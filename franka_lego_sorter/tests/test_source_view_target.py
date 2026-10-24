from pathlib import Path
import tempfile
import unittest
import numpy as np
from lego_sorter.commissioning import source_view_target


class SourceViewTargetTests(unittest.TestCase):
    def test_negative_raise_corridor_target_preserves_y_z_orientation(self):
        with tempfile.TemporaryDirectory()as d:
            path=Path(d)/'source.npz'
            source=np.eye(4);source[:3,3]=[.24,-.02,.348]
            np.savez(path,T_base_tcp=source,robot_drift_m=0.,robot_drift_rad=0.)
            start=source.copy();start[0,3]=.50;start[2,3]=.344
            target,intent=source_view_target(start,-.20,path)
            np.testing.assert_allclose(target[:3,3],[.30,-.02,.344])
            np.testing.assert_array_equal(target[:3,:3],start[:3,:3])
            self.assertEqual(intent['purpose'],'return_to_observed_source')
            self.assertEqual(len(intent['source_reference_sha256']),64)
            for distance in (0,.1,-.21,-.27):
                with self.subTest(distance=distance),self.assertRaises(ValueError):
                    source_view_target(start,distance,path)

    def test_low_or_unrelated_source_reference_refused(self):
        with tempfile.TemporaryDirectory()as d:
            path=Path(d)/'source.npz'
            source=np.eye(4);source[:3,3]=[.24,-.02,.30]
            np.savez(path,T_base_tcp=source,robot_drift_m=0.,robot_drift_rad=0.)
            start=source.copy();start[:3,3]=[.50,-.02,.344]
            with self.assertRaisesRegex(ValueError,'raised source'):
                source_view_target(start,-.20,path)


if __name__=='__main__':unittest.main()
