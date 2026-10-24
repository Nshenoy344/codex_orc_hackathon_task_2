import hashlib,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from validate_isolated_lift10 import contract

class CarryContractTests(unittest.TestCase):
    def fixture(self,d,xy_noise=.00000735):
        root=Path(d)
        def write(name,data):
            p=root/name;p.write_text(json.dumps(data));return p
        def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
        snapshot=write('source.npz',{});he=write('handeye.yaml',{})
        targets=write('targets.json',{'snapshot':str(snapshot),'snapshot_sha256':sha(snapshot),'handeye_artifact':str(he),'handeye_sha256':sha(he),'planes':[{'plane':'left'}]})
        gm={'targets':str(targets),'snapshot':str(snapshot),'handeye':str(he),'plane':'left'}
        gp=write('grasp_plan.json',{'isolated_diagnostic_stage':gm});gr=write('grasp_result.json',{})
        start=np.diag([1.,-1.,-1.,1.]);start[:3,3]=[0,-.5,.010]
        hold=write('hold.json',{'success':True,'status':'grasped_pending_lift','hand_grasp_and_body_width_verified':True,'brick_lifted_verified':False,'pose_plan':str(gp),'pose_plan_sha256':sha(gp),'pose_result':str(gr),'pose_result_sha256':sha(gr),'gripper_after':{'width_m':.01535},'T_base_tcp_before':start.tolist()})
        lp=write('lift_plan.json',{'isolated_diagnostic_stage':{'stage':'lift10','hold_result':str(hold)}})
        lr=write('lift_result.json',{'success':True,'isolated_lift10_result':{'success':True},'plan_file':str(lp),'T_base_tcp_after':start.tolist()})
        target=start.copy();target[2,3]=.300;target[0,3]+=xy_noise;target[1,3]+=.000000612;target[:3,:3]=Rotation.from_rotvec([0,0,.00001483]).as_matrix()@target[:3,:3]
        meta=dict(gm,stage='carry_observe300',hold_result=str(hold),lift10_result=str(lr),gripper_width_m=.01535,source_roi_px=[690,480,110,160])
        return write('plan.json',{'T_base_tcp_start':start.tolist(),'T_base_tcp_target':target.tolist(),'isolated_diagnostic_stage':meta})
    def test_two_pose_reads_micrometer_noise_is_not_lateral_intent(self):
        with tempfile.TemporaryDirectory()as d:contract(self.fixture(d))
    def test_ten_millimeter_lateral_move_is_rejected(self):
        with tempfile.TemporaryDirectory()as d:
            with self.assertRaises(ValueError):contract(self.fixture(d,xy_noise=.010))
    def test_absolute_carry_height_remains_exact(self):
        with tempfile.TemporaryDirectory()as d:
            p=self.fixture(d);data=json.loads(p.read_text());data['T_base_tcp_target'][2][3]=.300001;p.write_text(json.dumps(data))
            with self.assertRaises(ValueError):contract(p)

    def with_checkpoint(self,p,deflection=.002,start_extra=0.,width=.01535):
        data=json.loads(p.read_text());m=data['isolated_diagnostic_stage'];fresh=np.array(data['T_base_tcp_start']);fresh[0,3]+=deflection
        m['holding_checkpoint']={'T_base_tcp':fresh.tolist(),'width_m':width,'lift10_result_sha256':hashlib.sha256(Path(m['lift10_result']).read_bytes()).hexdigest()}
        data['T_base_tcp_start'][0][3]+=deflection+start_extra;data['T_base_tcp_target'][0][3]+=deflection+start_extra;p.write_text(json.dumps(data));return p

    def test_fresh_checkpoint_accepts_bounded2mm_holding_deflection(self):
        with tempfile.TemporaryDirectory()as d:contract(self.with_checkpoint(self.fixture(d)))

    def test_fresh_checkpoint_does_not_relax1mm_start_gate(self):
        with tempfile.TemporaryDirectory()as d:
            p=self.with_checkpoint(self.fixture(d),start_extra=.0011)
            with self.assertRaisesRegex(ValueError,'Arm moved'):contract(p)

    def test_checkpoint_cannot_exceed3mm_old_proof_envelope(self):
        with tempfile.TemporaryDirectory()as d:
            p=self.with_checkpoint(self.fixture(d),deflection=.0031)
            with self.assertRaisesRegex(ValueError,'exceeds3mm'):contract(p)

    def test_checkpoint_width_must_match_current_plan(self):
        with tempfile.TemporaryDirectory()as d:
            p=self.with_checkpoint(self.fixture(d),width=.014)
            with self.assertRaisesRegex(ValueError,'opening'):contract(p)

    def test_checkpoint_pins_immutable_lift_proof(self):
        with tempfile.TemporaryDirectory()as d:
            p=self.with_checkpoint(self.fixture(d));data=json.loads(p.read_text());data['isolated_diagnostic_stage']['holding_checkpoint']['lift10_result_sha256']='0'*64;p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'source changed'):contract(p)
