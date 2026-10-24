from setuptools import setup
from glob import glob
setup(name='franka_realsense_handeye', version='0.1.0',
      packages=['franka_realsense_handeye'],
      data_files=[('share/ament_index/resource_index/packages', ['resource/franka_realsense_handeye']),
                  ('share/franka_realsense_handeye', ['package.xml']),
                  ('share/franka_realsense_handeye/config', glob('config/*.yaml')),
                  ('share/franka_realsense_handeye/launch', glob('launch/*.py'))],
      install_requires=['setuptools'], zip_safe=True,
      maintainer='nshenoy', maintainer_email='nshenoy@users.noreply.github.com',
      description='Franka eye-in-hand ChArUco calibration for RealSense depth',
      license='Proprietary',
      entry_points={'console_scripts': [
          'calibrate = franka_realsense_handeye.node:main',
          'generate_board = franka_realsense_handeye.board:main']})
