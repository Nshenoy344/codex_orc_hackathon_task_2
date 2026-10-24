import numpy as np
import pytest
from lego_sorter.held_bottom_plane import bottom_plane_in_tcp, release_tcp_z


def test_measured_source_plane_and_tilted_bottom_clearance():
    T = np.eye(4); T[2, 3] = .016
    n, h = bottom_plane_in_tcp(T, [0, 0, 1], 0)
    points = np.array([[-.015,-.008,h],[.015,-.008,h],[.015,.008,h],[-.015,.008,h]])
    angle = .1
    R = np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    z = release_tcp_z([.3,.2], R, n, h, [0,0,1], .05, points, .003)
    heights = (points @ R.T)[:,2] + z
    assert min(heights) == pytest.approx(.053)
    assert z > .069


def test_refuses_invented_bottom_points():
    with pytest.raises(ValueError, match='lie on'):
        release_tcp_z([0,0], np.eye(3), [0,0,1], -.016, [0,0,1], .05,
                      [[0,0,0],[1,0,0],[0,1,0]], .003)
