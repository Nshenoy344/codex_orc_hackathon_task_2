"""Direct RealSense capture; depth is explicitly aligned to color pixels."""
from dataclasses import dataclass, field
import time
import os
import numpy as np


def limit_vision_threads():
    """Keep this process's OpenCV/TBB pool bounded beside the1kHz controller."""
    import cv2
    value=os.environ.get('LEGO_ANALYSIS_THREADS','1')
    if value not in ('1','2','3','4'):
        raise ValueError('LEGO_ANALYSIS_THREADS must be from1 through4')
    limit=int(value)
    if cv2.getNumThreads()!=limit:
        cv2.setNumThreads(limit)
    return limit


def configure_depth_units(sensor, sdk, requested=None):
    """Optional native Z16 scale experiment; finer units do not prove accuracy."""
    if requested is not None:
        value=float(requested)
        if not np.isfinite(value) or value<=0:
            raise ValueError('camera.depth_units_m must be finite and positive')
        option=sdk.option.depth_units
        if not sensor.supports(option):
            raise ValueError('Connected depth sensor does not support configurable depth units')
        bounds=sensor.get_option_range(option)
        if not float(bounds.min)<=value<=float(bounds.max):
            raise ValueError('Requested depth units are outside the device-reported option range')
        sensor.set_option(option,value)
    actual=float(sensor.get_depth_scale())
    if not np.isfinite(actual) or actual<=0:
        raise RuntimeError('Depth sensor returned an invalid actual depth scale')
    if requested is not None and not np.isclose(actual,float(requested),rtol=1e-4,atol=1e-9):
        raise RuntimeError('Depth sensor did not apply the requested depth-units setting')
    return actual


def configure_depth_options(sensor,sdk,config):
    """Apply explicit device-supported depth controls and record actual values."""
    options={'depth_visual_preset':'visual_preset','depth_emitter_enabled':'emitter_enabled','depth_laser_power':'laser_power'}
    pending=[];actual={}
    for key,name in options.items():
        option=getattr(sdk.option,name)
        if not sensor.supports(option):
            if config.get(key)is not None:raise ValueError('Unsupported camera option: '+key)
            continue
        if config.get(key)is not None:
            value=float(config[key]);bounds=sensor.get_option_range(option)
            if not np.isfinite(value)or not bounds.min<=value<=bounds.max:raise ValueError('Camera option outside device range: '+key)
            if bounds.step>0 and abs((value-bounds.min)/bounds.step-round((value-bounds.min)/bounds.step))>1e-5:raise ValueError('Camera option does not match device step: '+key)
            pending.append((key,option,value))
    for key,option,value in pending:
        sensor.set_option(option,value)
        if not np.isclose(sensor.get_option(option),value,atol=1e-5,rtol=0):raise RuntimeError('Depth camera option not applied: '+key)
    for key,name in options.items():
        option=getattr(sdk.option,name)
        if sensor.supports(option):actual[key]=float(sensor.get_option(option))
    return actual


@dataclass
class Frame:
    color_bgr: np.ndarray
    depth_m: np.ndarray
    intrinsics: dict
    captured_monotonic: float
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        limit_vision_threads()

    def deproject(self, u, v, depth):
        k = self.intrinsics
        if any(abs(c) > 1e-8 for c in k.get('coeffs', [])):
            import pyrealsense2 as rs
            intr = rs.intrinsics()
            for key in ('width', 'height', 'fx', 'fy', 'ppx', 'ppy', 'coeffs'):
                setattr(intr, key, k[key])
            intr.model = getattr(rs.distortion, k['distortion'].split('.')[-1])
            return np.array(rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], float(depth)))
        return np.array([(u-k['ppx']) / k['fx'] * depth,
                         (v-k['ppy']) / k['fy'] * depth, depth], dtype=float)


class RealSenseCamera:
    def __init__(self, config=None):
        self.config = config or {}
        self.pipeline = None
        self.started = False

    def __enter__(self):
        import pyrealsense2 as rs
        self.rs = rs
        cfg = rs.config()
        if self.config.get('serial'):
            cfg.enable_device(str(self.config['serial']))
        width, height, fps = (int(self.config.get(k, d))
                              for k, d in [('width', 640), ('height', 480), ('fps', 30)])
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.pipeline = rs.pipeline()
        try:
            # Advanced stereo presets may reject UVC control writes once streams
            # are active. Resolve the requested device before starting them.
            if any(self.config.get(k)is not None for k in ('depth_visual_preset','depth_emitter_enabled','depth_laser_power')):
                resolved=cfg.resolve(rs.pipeline_wrapper(self.pipeline))
                configure_depth_options(resolved.get_device().first_depth_sensor(),rs,self.config)
            profile = self.pipeline.start(cfg)
            self.started = True
            depth_sensor=profile.get_device().first_depth_sensor()
            depth_options=configure_depth_options(depth_sensor,rs,{})
            for key,value in depth_options.items():
                if self.config.get(key)is not None and not np.isclose(value,float(self.config[key]),atol=1e-5,rtol=0):raise RuntimeError('Depth camera option changed at stream startup: '+key)
            self.depth_scale = configure_depth_units(depth_sensor,rs,self.config.get('depth_units_m'))
            self.align = rs.align(rs.stream.color)
            # SDK factory extrinsics map native depth points into color optical
            # coordinates. Build columns using transformed basis points rather
            # than relying on a remembered matrix storage order.
            extrinsics=profile.get_stream(rs.stream.depth).get_extrinsics_to(profile.get_stream(rs.stream.color))
            origin=np.asarray(rs.rs2_transform_point_to_point(extrinsics,[0.,0.,0.]))
            self.T_color_depth=np.eye(4)
            self.T_color_depth[:3,3]=origin
            for axis in range(3):
                basis=[0.,0.,0.];basis[axis]=1.
                self.T_color_depth[:3,axis]=np.asarray(rs.rs2_transform_point_to_point(extrinsics,basis))-origin
            rotation=self.T_color_depth[:3,:3]
            if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(rotation),1.,atol=1e-5):
                raise RuntimeError('SDK returned invalid depth-to-color factory extrinsics')
            color_sensor = profile.get_device().first_color_sensor()
            # Disable auto controls only when explicit measured settings are supplied.
            for name, auto_name in [('exposure', 'enable_auto_exposure'),
                                    ('white_balance', 'enable_auto_white_balance')]:
                if self.config.get(name) is not None:
                    auto, option = getattr(rs.option, auto_name), getattr(rs.option, name)
                    if color_sensor.supports(auto):
                        color_sensor.set_option(auto, 0)
                    if color_sensor.supports(option):
                        color_sensor.set_option(option, float(self.config[name]))
            device = profile.get_device()
            self.metadata = dict(serial=device.get_info(rs.camera_info.serial_number),
                                 camera_name=device.get_info(rs.camera_info.name),
                                 width=width, height=height, fps=fps)
            self.metadata.update(depth_options)
            native_depth=profile.get_stream(rs.stream.depth).as_video_stream_profile()
            self.metadata.update(depth_units_m=self.depth_scale,
                native_depth_profile=dict(width=native_depth.width(),height=native_depth.height(),fps=native_depth.fps(),format=str(native_depth.format())))
            for key in ('enable_auto_exposure', 'enable_auto_white_balance'):
                option = getattr(rs.option, key)
                self.metadata[key] = color_sensor.get_option(option) if color_sensor.supports(option) else None
            for key, auto_key in [('exposure', 'enable_auto_exposure'), ('white_balance', 'enable_auto_white_balance')]:
                option = getattr(rs.option, key)
                self.metadata[key] = (color_sensor.get_option(option)
                                      if not self.metadata[auto_key] and color_sensor.supports(option) else 'auto')
            for _ in range(int(self.config.get('warmup_frames', 30))):
                self.pipeline.wait_for_frames(10000)
        except Exception:
            self.close()
            raise
        return self

    def capture(self):
        if self.pipeline is None:
            raise RuntimeError('Camera is not open')
        frames = self.align.process(self.pipeline.wait_for_frames(10000))
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color or not depth:
            raise RuntimeError('RealSense did not produce aligned color and depth')
        intr = depth.profile.as_video_stream_profile().intrinsics
        if intr.model not in (self.rs.distortion.none, self.rs.distortion.brown_conrady,
                              self.rs.distortion.modified_brown_conrady,
                              self.rs.distortion.inverse_brown_conrady):
            raise RuntimeError(f'Unsupported aligned intrinsics distortion: {intr.model}')
        # Retain aligned color distortion for SDK-aware deprojection in Frame.
        color_image = np.asanyarray(color.get_data()).copy()
        depth_image = np.asanyarray(depth.get_data()).astype(np.float32) * self.depth_scale
        k = dict(fx=intr.fx, fy=intr.fy, ppx=intr.ppx, ppy=intr.ppy,
                 width=intr.width, height=intr.height,
                 distortion=str(intr.model), coeffs=list(intr.coeffs))
        if intr.model == self.rs.distortion.modified_brown_conrady and any(abs(c) > 1e-8 for c in intr.coeffs):
            raise RuntimeError('Modified Brown distortion cannot be directly deprojected by RealSense; use rectified color.')
        return Frame(color_image, depth_image, k, time.monotonic(), dict(self.metadata))

    def capture_burst(self, count=None):
        count = int(count or self.config.get('burst_frames', 5))
        if count < 1:
            raise ValueError('burst_frames must be positive')
        frames = [self.capture() for _ in range(count)]
        depths = np.stack([f.depth_m for f in frames])
        depths[depths <= 0] = np.nan
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            depth = np.nanmedian(depths, axis=0)
        depth[~np.isfinite(depth)] = 0
        colors = np.stack([f.color_bgr for f in frames])
        # A stationary burst also rejects a moving scene rather than blending it.
        motion = float(np.median(np.abs(colors[0].astype(float)-colors[-1].astype(float))))
        if motion > float(self.config.get('max_burst_motion_intensity', 8.0)):
            raise RuntimeError(f'Scene moved during stationary capture ({motion:.1f})')
        return Frame(np.median(colors, axis=0).astype(np.uint8), depth,
                     frames[-1].intrinsics, frames[-1].captured_monotonic, frames[-1].metadata)

    def close(self):
        if self.pipeline is not None:
            try:
                if self.started:self.pipeline.stop()
            finally:
                self.pipeline = None
                self.started = False

    def __exit__(self, *_):
        self.close()
