import unittest
import cv2
import numpy as np
from lego_sorter.charuco_observation import _core
from lego_sorter.wrist_stereo import solve,transform

class WristStereoTests(unittest.TestCase):
    def data(self):
        config=dict(dictionary='DICT_4X4_1000',squares_x=7,squares_y=5,square_length_m=.03,marker_length_m=.021,legacy_pattern=False)
        board,_=_core().make_board(config);points=board.chessboardCorners
        kr=np.array([[920.,0,640],[0,915.,360],[0,0,1]])
        kw=np.array([[890.,0,633],[0,900.,357],[0,0,1]])
        dw=np.array([-.08,.02,.0002,-.0001,0.])
        stereo=transform(cv2.Rodrigues(np.array([.02,-.03,3.10]))[0],[.08,.025,-.01])
        samples=[]
        for i in range(12):
            rv=np.array([.32*np.sin(i*1.2),.32*np.cos(i*.9),.1*np.sin(i*.5)])
            br=transform(cv2.Rodrigues(rv)[0],[-.09+.03*np.sin(i),-.065+.02*np.cos(i),.42+.06*np.sin(i*.7)])
            wb=stereo@br
            rp=cv2.projectPoints(points,rv,br[:3,3],kr,np.zeros(5))[0].reshape(-1,2)
            wp=cv2.projectPoints(points,cv2.Rodrigues(wb[:3,:3])[0],wb[:3,3],kw,dw)[0].reshape(-1,2)
            samples.append(dict(rs_corners={str(j):p.tolist()for j,p in enumerate(rp)},webcam_corners={str(j):p.tolist()for j,p in enumerate(wp)}))
        return samples,config,kr,stereo,kw
    def test_known_lens_stereo_and_tcp_composition(self):
        samples,cfg,kr,stereo,kw=self.data();tcp=transform(np.eye(3),[.05,-.03,-.04])
        r=solve(samples,cfg,[1280,720],kr,np.zeros(5),tcp,np.eye(4))
        self.assertTrue(r['accepted'],r['checks'])
        np.testing.assert_allclose(r['webcam_camera_matrix'],kw,atol=.02)
        np.testing.assert_allclose(r['T_logitech_realsense_color'],stereo,atol=2e-5)
        np.testing.assert_allclose(r['T_tcp_logitech'],tcp@np.linalg.inv(stereo),atol=2e-5)
        self.assertFalse(r['physical_commissioning_verified'])
    def test_held_out_error_rejects_solution(self):
        samples,cfg,kr,_,_=self.data()
        for p in samples[3]['webcam_corners'].values():p[0]+=30
        r=solve(samples,cfg,[1280,720],kr,np.zeros(5),np.eye(4))
        self.assertFalse(r['accepted']);self.assertFalse(r['checks']['held_out_reprojection'])
    def test_single_planar_view_cannot_be_claimed_calibrated(self):
        samples,cfg,kr,_,_=self.data()
        with self.assertRaisesRegex(ValueError,'at least8'):solve(samples[:1],cfg,[1280,720],kr,np.zeros(5),np.eye(4))

if __name__=='__main__':unittest.main()
