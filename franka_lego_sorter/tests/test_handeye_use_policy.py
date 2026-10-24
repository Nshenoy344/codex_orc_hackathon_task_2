from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import yaml
from lego_sorter.cli import load_handeye, load_handeye_candidate, main


class HandeyeUsePolicyTests(unittest.TestCase):
    def artifact(self, path, *, physical=False, accepted=True):
        value=dict(parameters=dict(base_frame='fr3_link0',tcp_frame='fr3_hand_tcp'),
                   T_tcp_color=np.eye(4).tolist(),physical_commissioning_verified=physical,
                   diagnostics=dict(accepted=accepted))
        path.write_text(yaml.safe_dump(value))
        return dict(handeye_file=str(path),motion=dict(base_frame='fr3_link0',tcp_frame='fr3_hand_tcp'))

    def test_full_sorting_rejects_accepted_but_physically_unverified_candidate(self):
        with tempfile.TemporaryDirectory() as folder:
            config=self.artifact(Path(folder)/'candidate.yaml')
            with self.assertRaisesRegex(ValueError,'not physically commissioned'):
                load_handeye(config)
            np.testing.assert_allclose(load_handeye_candidate(config),np.eye(4))

    def test_explicit_candidate_path_rejects_numerically_rejected_pose(self):
        with tempfile.TemporaryDirectory() as folder:
            config=self.artifact(Path(folder)/'rejected.yaml',accepted=False)
            with self.assertRaisesRegex(ValueError,'Rejected numerical'):
                load_handeye_candidate(config)

    def test_physical_flag_must_be_a_real_boolean_not_a_string(self):
        with tempfile.TemporaryDirectory() as folder:
            config=self.artifact(Path(folder)/'candidate.yaml',physical='true')
            with self.assertRaisesRegex(ValueError,'not physically commissioned'):
                load_handeye(config)

    def test_physically_verified_calibration_can_be_loaded_normally(self):
        with tempfile.TemporaryDirectory() as folder:
            config=self.artifact(Path(folder)/'verified.yaml',physical=True)
            np.testing.assert_allclose(load_handeye(config),np.eye(4))

    def test_explicit_candidate_path_still_checks_tcp_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            config=self.artifact(Path(folder)/'candidate.yaml')
            config['motion']['tcp_frame']='different_tcp'
            with self.assertRaisesRegex(ValueError,'tcp_frame differs'):
                load_handeye_candidate(config)

    def test_candidate_cli_is_read_only_and_never_loads_motion_adapter(self):
        with tempfile.TemporaryDirectory() as folder:
            config=self.artifact(Path(folder)/'candidate.yaml')
            with patch('lego_sorter.cli.load_config',return_value=config),redirect_stdout(io.StringIO()) as output:
                result=main(['inspect-calibration-candidate'])
            self.assertEqual(result,0)
            self.assertIn('"full_sorting_authorized_by_this_command": false',output.getvalue())

    def test_full_run_has_no_candidate_bypass_before_ros_startup(self):
        with tempfile.TemporaryDirectory() as folder:
            config=self.artifact(Path(folder)/'candidate.yaml')
            with patch('lego_sorter.cli.load_config',return_value=config),patch('lego_sorter.cli.load_poses',return_value={}):
                with self.assertRaisesRegex(ValueError,'not physically commissioned'):
                    main(['run','--execute','--max-transfers','1'])


if __name__=='__main__':unittest.main()
