"""Standalone RealSense color/depth preview. Escape or Q closes the window."""
import logging
import sys
import cv2
import numpy as np
import pyrealsense2 as rs

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def main():
    devices = rs.context().query_devices()
    if not devices:
        raise RuntimeError('No RealSense camera detected. Check USB connection and device permissions.')
    for device in devices:
        logging.info('Camera: %s, serial: %s', device.get_info(rs.camera_info.name),
                     device.get_info(rs.camera_info.serial_number))
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    started = False
    title = 'RealSense: color | depth (Q or Escape to close)'
    try:
        pipeline.start(config)
        started = True
        colorizer = rs.colorizer()
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        logging.info('Streaming color and depth at 640x480, 30 FPS')
        while True:
            frames = pipeline.wait_for_frames(10000)
            color, depth = frames.get_color_frame(), frames.get_depth_frame()
            if not color or not depth:
                continue
            rgb = np.asanyarray(color.get_data())
            depth_rgb = np.asanyarray(colorizer.colorize(depth).get_data())
            depth_bgr = cv2.cvtColor(depth_rgb, cv2.COLOR_RGB2BGR)
            if depth_bgr.shape[:2] != rgb.shape[:2]:
                depth_bgr = cv2.resize(depth_bgr, (rgb.shape[1], rgb.shape[0]))
            cv2.imshow(title, np.hstack((rgb, depth_bgr)))
            if cv2.waitKey(1) & 0xff in (27, ord('q')):
                break
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if started:
            pipeline.stop()
        cv2.destroyAllWindows()
        logging.info('Viewer stopped')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception:
        logging.exception('Camera viewer failed')
        sys.exit(1)
