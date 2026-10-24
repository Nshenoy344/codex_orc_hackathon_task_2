import importlib.util
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from validate_first_loose_release_pose import command_matches_expected


def test_fresh_sample_jitter_bounded_without_accepting_wrong_command():
    expected=np.eye(4)
    command=expected.copy();command[0,3]=.00004
    command[:3,:3]=Rotation.from_rotvec([0,0,.00008]).as_matrix()
    assert command_matches_expected(command,expected)
    command[0,3]=.001
    assert not command_matches_expected(command,expected)
    command[0,3]=0
    command[:3,:3]=Rotation.from_rotvec([0,0,.00011]).as_matrix()
    assert not command_matches_expected(command,expected)
