import hashlib,json,tempfile,unittest
from pathlib import Path
from lego_sorter.review_cache import verify_view_shift_review

class ViewReviewCacheTests(unittest.TestCase):
    def test_all_pins_and_changed_file_rejection(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);(root/'tools').mkdir();(root/'data/commissioning').mkdir(parents=True)
            paths={'snapshot_sha256':root/'snapshot.npz','handeye_sha256':root/'handeye.yaml','geometry_sha256':root/'geometry.json','wrapper_sha256':root/'tools/validate_view_shift.py','reviewer_sha256':root/'tools/validate_isolated_hover.py','urdf_sha256':root/'data/commissioning/fr3_stock_hand.urdf'}
            for path in paths.values():path.write_bytes(b'pinned input')
            plan=root/'plan.json';plan.write_text(json.dumps({'commissioning_view_shift':{'snapshot':str(paths['snapshot_sha256']),'handeye':str(paths['handeye_sha256']),'gripper_width_m':.02}}));paths['plan_sha256']=plan
            checks={name:True for name in ['application_wrist_branch','stock_hand_cloud_clear','full_arm_cloud_clear','full_arm_table_clear','camera_bracket_assumed_envelope_clear','minimum150mm_tip_clearance','minimum8s_duration','viewing_line','viewing_orientation','viewing_progress']}
            review={'purpose':'noncontact_view_shift','passed_numeric_checks':True,'checks':checks,'failed_checks':[],'snapshot':str(paths['snapshot_sha256']),'handeye':str(paths['handeye_sha256']),'geometry_report':str(paths['geometry_sha256']),'hand_opening_assumed_m':.02}
            review.update({key:hashlib.sha256(path.read_bytes()).hexdigest()for key,path in paths.items()})
            self.assertTrue(verify_view_shift_review(plan,review,root))
            for key,path in paths.items():
                with self.subTest(changed=key):
                    before=path.read_bytes();path.write_bytes(before+b'\n')
                    with self.assertRaisesRegex(ValueError,'changed'):verify_view_shift_review(plan,review,root)
                    path.write_bytes(before)
            review['checks']['viewing_line']=False
            with self.assertRaisesRegex(ValueError,'passing'):verify_view_shift_review(plan,review,root)
