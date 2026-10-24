"""Service-driven capture; this node never commands robot motion."""
from pathlib import Path
import json
import os
import tempfile
import cv2
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from rclpy.executors import ExternalShutdownException
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Image, CameraInfo
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener
from .board import make_board, detect
from .core import transform, rotation_deg, solve


def tf_matrix(msg):
    t, q = msg.transform.translation, msg.transform.rotation
    q = np.array([q.x, q.y, q.z, q.w], dtype=float)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-9:
        raise ValueError('Invalid TF quaternion')
    x, y, z, w = q / norm
    rot = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                    [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                    [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    return transform(rot, [t.x, t.y, t.z])


def atomic_yaml(path, data):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
            name = stream.name
            yaml.safe_dump(data, stream, sort_keys=False)
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


class Calibrator(Node):
    def __init__(self):
        super().__init__('handeye_calibrator')
        defaults = dict(base_frame='fr3_link0', tcp_frame='fr3_hand_tcp',
                        color_frame='camera_color_optical_frame',
                        depth_frame='camera_depth_optical_frame',
                        image_topic='/camera/camera/color/image_raw',
                        camera_info_topic='/camera/camera/color/camera_info',
                        dictionary='DICT_5X5_1000', squares_x=12, squares_y=7,
                        square_length_m=0.04, marker_length_m=0.03, legacy_pattern=False,
                        min_charuco_corners=20, max_reprojection_error_px=0.9,
                        min_samples=12, method='PARK', max_image_age_sec=1.0,
                        tf_timeout_sec=0.2, min_pose_translation_m=0.015,
                        min_pose_rotation_deg=5.0, min_rotation_span_deg=15.0,
                        max_board_translation_rms_m=0.01, max_board_rotation_rms_deg=2.0,
                        output_file='~/franka_handeye_results/calibration.yaml')
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        self.p = {key: self.get_parameter(key).value for key in defaults}
        for key in ('base_frame', 'tcp_frame', 'color_frame', 'depth_frame'):
            if not self.p[key] or self.p[key].startswith('/'):
                raise ValueError(f'{key} must be a nonempty TF frame without leading slash')
        for key, value in self.p.items():
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(f'{key} must be finite and positive')
        if self.p['min_samples'] < 3 or self.p['min_charuco_corners'] < 4:
            raise ValueError('Need >=3 samples and >=4 corners')
        if self.p['method'] not in ('TSAI', 'PARK', 'HORAUD', 'ANDREFF', 'DANIILIDIS'):
            raise ValueError('Unknown hand-eye method')
        self.board, self.dictionary = make_board(self.p)
        self.add_on_set_parameters_callback(lambda _: SetParametersResult(
            successful=False, reason='Restart with edited YAML to keep capture settings consistent'))
        self.buffer = Buffer(cache_time=Duration(seconds=30.0))
        self.tf_node = Node('handeye_tf_listener')
        self.listener = TransformListener(self.buffer, self.tf_node, spin_thread=True)
        self.bridge = CvBridge()
        self.image = self.info = None
        self.samples = []
        self.create_subscription(Image, self.p['image_topic'], self.on_image, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, self.p['camera_info_topic'], self.on_info,
                                 qos_profile_sensor_data)
        for name, callback in [('capture_sample', self.capture), ('solve', self.calibrate),
                               ('reset', self.reset), ('status', self.status)]:
            self.create_service(Trigger, '~/'+name, self.guarded(callback))
        self.get_logger().info('Ready: move to a stationary pose, call ~/capture_sample; then ~/solve')

    def destroy_node(self):
        self.listener.unregister()
        self.listener.executor.shutdown()
        self.tf_node.destroy_node()
        return super().destroy_node()

    def guarded(self, fn):
        def callback(request, response):
            try:
                response.message = fn()
                response.success = True
            except Exception as exc:
                response.success = False
                response.message = str(exc)
                self.get_logger().warning(str(exc))
            return response
        return callback

    def on_image(self, msg):
        self.image = msg

    def on_info(self, msg):
        self.info = msg

    def lookup(self, parent, child, stamp):
        return tf_matrix(self.buffer.lookup_transform(
            parent, child, stamp, timeout=Duration(seconds=self.p['tf_timeout_sec'])))

    def capture(self):
        image, info = self.image, self.info
        if image is None or info is None:
            raise ValueError('Waiting for color Image and CameraInfo')
        stamp = Time.from_msg(image.header.stamp)
        if stamp.nanoseconds == 0:
            raise ValueError('Image timestamp is zero; exact-time TF is required')
        age = (self.get_clock().now() - stamp).nanoseconds / 1e9
        if not 0 <= age <= self.p['max_image_age_sec']:
            raise ValueError(f'Image is stale or in the future ({age:.3f}s); check clocks')
        if image.header.frame_id != self.p['color_frame'] or info.header.frame_id != self.p['color_frame']:
            raise ValueError('Image/CameraInfo frame must match color_frame')
        if (image.width, image.height) != (info.width, info.height):
            raise ValueError('CameraInfo dimensions do not match image')
        if info.distortion_model not in ('plumb_bob', 'rational_polynomial'):
            raise ValueError('Requires raw pinhole color images with plumb_bob/rational_polynomial intrinsics')
        k = np.array(info.k).reshape(3, 3)
        d = np.array(info.d, dtype=float)
        if not np.isfinite(k).all() or not np.isfinite(d).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
            raise ValueError('Invalid camera intrinsics')
        signature = dict(k=list(info.k), d=list(info.d), width=info.width,
                         height=info.height, distortion_model=info.distortion_model)
        if self.samples and signature != self.samples[0]['intrinsics']:
            raise ValueError('Intrinsics changed during session; reset before continuing')
        if any(s['stamp_ns'] == stamp.nanoseconds for s in self.samples):
            raise ValueError('This image was already captured')
        robot = self.lookup(self.p['base_frame'], self.p['tcp_frame'], stamp)
        color_depth = self.lookup(self.p['color_frame'], self.p['depth_frame'], stamp)
        for s in self.samples:
            prior = np.array(s['T_base_tcp'])
            if (np.linalg.norm(robot[:3, 3] - prior[:3, 3]) < self.p['min_pose_translation_m'] and
                    rotation_deg(prior[:3, :3].T @ robot[:3, :3]) < self.p['min_pose_rotation_deg']):
                raise ValueError('Pose too close to an existing sample')
        rv, tv, rms, count = detect(self.bridge.imgmsg_to_cv2(image, 'bgr8'), k, d,
                                    self.board, self.dictionary, self.p['min_charuco_corners'])
        if not np.isfinite(rms) or rms > self.p['max_reprojection_error_px']:
            raise ValueError(f'Reprojection RMS too high: {rms:.3f} pixels')
        sample = dict(stamp_ns=stamp.nanoseconds, T_base_tcp=robot.tolist(),
                      T_color_board=transform(cv2.Rodrigues(rv)[0], tv).tolist(),
                      T_color_depth=color_depth.tolist(), reprojection_rms_px=rms,
                      corners=count, intrinsics=signature)
        # Persist before accepting a sample, so disk failures cannot silently lose it.
        atomic_yaml(self.p['output_file'] + '.samples.yaml',
                    dict(parameters=self.p, samples=self.samples + [sample]))
        self.samples.append(sample)
        return f'Captured {len(self.samples)} samples; RMS={rms:.3f}px, corners={count}'

    def calibrate(self):
        result, metrics = solve([s['T_base_tcp'] for s in self.samples],
                                [s['T_color_board'] for s in self.samples], self.p['method'],
                                self.p['min_samples'], self.p['min_rotation_span_deg'])
        if (metrics['board_translation_rms_m'] > self.p['max_board_translation_rms_m'] or
                metrics['board_rotation_rms_deg'] > self.p['max_board_rotation_rms_deg']):
            raise ValueError(f'Board consistency check failed: {metrics}; inspect samples and recapture')
        extrinsics = np.array([s['T_color_depth'] for s in self.samples])
        if not np.allclose(extrinsics, extrinsics[0], atol=1e-6):
            raise ValueError('RealSense color/depth extrinsics changed during capture')
        tcp_depth = result @ extrinsics[0]
        output = dict(convention='T_A_B maps B coordinates into A; meters; column vectors',
                      parameters=self.p, sample_count=len(self.samples), metrics=metrics,
                      T_tcp_color=result.tolist(), T_tcp_depth=tcp_depth.tolist(),
                      T_depth_tcp=np.linalg.inv(tcp_depth).tolist(),
                      T_color_depth=extrinsics[0].tolist(), samples=self.samples)
        atomic_yaml(self.p['output_file'], output)
        return json.dumps(dict(file=str(Path(self.p['output_file']).expanduser()),
                               T_tcp_depth=tcp_depth.tolist(), metrics=metrics))

    def reset(self):
        atomic_yaml(self.p['output_file'] + '.samples.yaml', dict(parameters=self.p, samples=[]))
        self.samples.clear()
        return 'Samples cleared; previously solved calibration file retained'

    def status(self):
        return f'Samples: {len(self.samples)}; required: {self.p["min_samples"]}'


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = Calibrator()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()
