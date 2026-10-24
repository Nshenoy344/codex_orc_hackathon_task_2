import importlib.util
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('translation_validator',Path(__file__).resolve().parents[1]/'tools/validate_translation_plan.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def scene(points=None,duration=20.,upper=.5,extra_y=0.):
    urdf=f'''<robot name="test"><link name="base"/><link name="xlink"/><link name="tcp"/>
    <joint name="x" type="prismatic"><parent link="base"/><child link="xlink"/><origin xyz="0 0 .35"/><axis xyz="1 0 0"/><limit lower="-.1" upper="{upper}" velocity=".2" effort="1"/></joint>
    <joint name="y" type="prismatic"><parent link="xlink"/><child link="tcp"/><axis xyz="0 1 0"/><limit lower="-.1" upper=".1" velocity=".2" effort="1"/></joint></robot>'''
    start=np.eye(4);start[2,3]=.35;target=start.copy();target[0,3]=.2
    if points is None:points=[(0.,0.),(.2,duration)]
    data=dict(tcp_frame='tcp',base_frame='base',joint_names=['x','y'],fraction=1.,
        T_base_tcp_start=start.tolist(),T_base_tcp_target=target.tolist(),
        points=[dict(positions=[q,extra_y*(t/duration)],velocities=[0.,0.],accelerations=[0.,0.],time_sec=t) for q,t in points])
    return urdf,data


class TranslationPlanTests(unittest.TestCase):
    def validate(self,**kwargs):
        urdf,data=scene(**kwargs)
        with tempfile.TemporaryDirectory() as d:
            u=Path(d)/'robot.urdf';p=Path(d)/'plan.json';u.write_text(urdf);p.write_text(json.dumps(data))
            return module.validate(p,u)

    def negative_review(self,source_x=.24,points=None,mutate_snapshot=False):
        urdf,data=scene(points=points or [(0.,0.),(-.2,20.)])
        urdf=urdf.replace('origin xyz="0 0 .35"','origin xyz=".45 0 .35"').replace('lower="-.1"','lower="-.3"',1)
        start=np.array(data['T_base_tcp_start']);start[0,3]=.45
        target=start.copy();target[0,3]-=.2
        data['T_base_tcp_start']=start.tolist();data['T_base_tcp_target']=target.tolist()
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);u=root/'robot.urdf';p=root/'plan.json';ref=root/'source.npz'
            source=start.copy();source[0,3]=source_x
            np.savez(ref,T_base_tcp=source,robot_drift_m=0.,robot_drift_rad=0.)
            data['commissioning_translation']=dict(axis='x',distance_m=-.2,purpose='return_to_observed_source',
                source_reference_snapshot=str(ref),source_reference_sha256=hashlib.sha256(ref.read_bytes()).hexdigest(),
                T_base_tcp_source_reference=source.tolist())
            if mutate_snapshot:
                np.savez(ref,T_base_tcp=source,robot_drift_m=.0001,robot_drift_rad=0.)
            u.write_text(urdf);p.write_text(json.dumps(data))
            return module.validate(p,u)

    def test_negative_source_return_preserves_all_dense_checks(self):
        report=self.negative_review()
        self.assertTrue(report['passed_numeric_checks'])
        self.assertAlmostEqual(report['endpoint_displacement_m'][0],-.2)
        self.assertGreaterEqual(report['min_incremental_directed_progress_m'],-1e-7)
        self.assertEqual(report['observed_source_return_policy']['purpose'],'return_to_observed_source')
        self.assertGreaterEqual(report['minimum_tcp_z_m'],.33)

    def test_negative_source_return_cannot_reverse_direction_or_overshoot_source(self):
        report=self.negative_review(points=[(0.,0.),(-.15,7.),(-.1,13.),(-.2,20.)])
        self.assertFalse(report['passed_numeric_checks'])
        with self.assertRaisesRegex(ValueError,'approach'):
            self.negative_review(source_x=.30)

    def test_negative_source_snapshot_must_remain_hash_pinned(self):
        with self.assertRaisesRegex(ValueError,'snapshot changed'):
            self.negative_review(mutate_snapshot=True)

    def test_exact_slow_x_shift_passes(self):
        report=self.validate()
        self.assertTrue(report['passed_numeric_checks'])
        self.assertAlmostEqual(report['endpoint_displacement_m'][0],.2)
        self.assertGreaterEqual(report['samples_checked'],2001)
        self.assertFalse(report['camera_collision_modeled'])

    def test_fresh_transform_rotation_jitter_tolerated(self):
        from scipy.spatial.transform import Rotation
        urdf,data=scene()
        data['T_base_tcp_target'][:3]  # retain the measured translation
        target=np.array(data['T_base_tcp_target'])
        target[:3,:3]=Rotation.from_rotvec([0.,0.,1e-6]).as_matrix()
        data['T_base_tcp_target']=target.tolist()
        with tempfile.TemporaryDirectory() as d:
            u=Path(d)/'robot.urdf';p=Path(d)/'plan.json'
            u.write_text(urdf);p.write_text(json.dumps(data))
            report=module.validate(p,u)
        self.assertTrue(report['passed_numeric_checks'])
        self.assertAlmostEqual(report['requested_orientation_change_rad'],1e-6)
        self.assertLessEqual(report['max_orientation_change_rad'],.002)

    def test_cross_axis_drift_rejected(self):
        self.assertFalse(self.validate(extra_y=.001)['passed_numeric_checks'])

    def test_short_duration_rejected(self):
        self.assertFalse(self.validate(duration=8.)['passed_numeric_checks'])

    def test_joint_bound_rejected(self):
        report=self.validate(upper=.1)
        self.assertFalse(report['passed_numeric_checks'])
        self.assertFalse(report['joint_limits']['x']['passed'])

    def test_nonmonotonic_x_rejected(self):
        report=self.validate(points=[(0.,0.),(.15,7.),(.1,13.),(.2,20.)])
        self.assertFalse(report['passed_numeric_checks'])
        self.assertLess(report['min_incremental_base_x_m'],0.)


if __name__=='__main__':unittest.main()
