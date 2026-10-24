import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from lego_sorter.held_view_checkpoint import build_checkpoint,verify_checkpoint

class HeldViewCheckpointTests(unittest.TestCase):
    def fixture(self,d,reason='Pose verification failed:0.0024m,0.006rad'):
        root=Path(d)
        def write(n,data):p=root/n;p.write_text(json.dumps(data));return p
        hold=write('hold.json',{'success':True,'hand_grasp_and_body_width_verified':True,'gripper_after':{'width_m':.01535}});proof=write('proof.json',{'combined_pick_evidence':True});he=write('handeye.yaml',{});targets=write('targets.json',{});env=write('environment.json',{})
        actual=np.eye(4);actual[2,3]=.4;goal=actual.copy();goal[0,3]=.0024
        plan=write('plan.json',{'T_base_tcp_target':goal.tolist(),'isolated_diagnostic_stage':{'stage':'carry_template_view','hold_result':str(hold),'source_proof':str(proof),'handeye':str(he),'targets':str(targets),'environment_boxes':str(env)}})
        result=write('result.json',{'success':False,'status':'execution_or_verification_failed','reason':reason,'plan_file':str(plan),'T_base_tcp_after':actual.tolist()});snapshot=write('capture.npz',{})
        return snapshot,result,actual
    def test_measured_checkpoint_does_not_rewrite_failed_motion(self):
        with tempfile.TemporaryDirectory()as d:
            snapshot,result,actual=self.fixture(d)
            with patch('lego_sorter.held_view_checkpoint.load_frame',return_value=(None,actual)):
                c=build_checkpoint(snapshot,result,actual,{'width_m':.01535})
            self.assertFalse(c['motion_success_claimed']);self.assertFalse(c['associated_execution_result_success']);self.assertFalse(json.loads(result.read_text())['success']);self.assertAlmostEqual(c['associated_target_position_error_m'],.0024)
            out=Path(d)/'checkpoint.json';out.write_text(json.dumps(c));verify_checkpoint(out,actual,.01535)
            changed=actual.copy();changed[0,3]+=.0011
            with self.assertRaisesRegex(ValueError,'1mm'):verify_checkpoint(out,changed,.01535)
            Path(c['targets']).write_text('{}changed')
            with self.assertRaisesRegex(ValueError,'changed'):verify_checkpoint(out)
    def test_unknown_timeout_cannot_be_reclassified_as_endpoint_checkpoint(self):
        with tempfile.TemporaryDirectory()as d:
            snapshot,result,actual=self.fixture(d,reason='ROS response timeout after60s')
            with self.assertRaisesRegex(ValueError,'unknown execution'):build_checkpoint(snapshot,result,actual,{'width_m':.01535})
    def test_lost_holding_width_rejected(self):
        with tempfile.TemporaryDirectory()as d:
            snapshot,result,actual=self.fixture(d)
            with patch('lego_sorter.held_view_checkpoint.load_frame',return_value=(None,actual)):
                with self.assertRaisesRegex(ValueError,'hold'):build_checkpoint(snapshot,result,actual,{'width_m':.001})
