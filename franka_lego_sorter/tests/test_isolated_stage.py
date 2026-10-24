import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from validate_isolated_stage import review
from lego_sorter.review_cache import verify_isolated_approach_review

class IsolatedStageTests(unittest.TestCase):
    def test_contact_stage_cannot_enter_noncontact_validator(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'plan.json';p.write_text(json.dumps({'isolated_diagnostic_stage':{'stage':'grasp'}}))
            with self.assertRaisesRegex(ValueError,'Only noncontact approach30'):review(p)
    def test_cache_rejects_missing_scene_checks(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'plan.json';p.write_text(json.dumps({'isolated_diagnostic_stage':{'stage':'approach30'}}))
            with self.assertRaisesRegex(ValueError,'complete passing'):verify_isolated_approach_review(p,{'purpose':'isolated_approach30','passed_numeric_checks':True})
