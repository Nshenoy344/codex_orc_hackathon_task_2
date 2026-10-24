"""Read one fresh published Franka state; no actions/services/FCI connection."""
import json,time
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation
from franka_msgs.msg import FrankaRobotState

def pose(value):
    p=value.pose;t=np.eye(4);t[:3,3]=[p.position.x,p.position.y,p.position.z]
    t[:3,:3]=Rotation.from_quat([p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w]).as_matrix()
    return t.ravel(order='F').tolist()
rclpy.init();n=rclpy.create_node('wrist_calibration_read_only_state');messages=[]
sub=n.create_subscription(FrankaRobotState,'/franka_robot_state_broadcaster/robot_state',messages.append,qos_profile_sensor_data)
try:
    start=time.monotonic()
    while not messages and time.monotonic()-start<6:rclpy.spin_once(n,timeout_sec=.1)
    if not messages:raise RuntimeError('No fresh published robot state')
    s=messages[-1];age=(n.get_clock().now().nanoseconds-(s.header.stamp.sec*10**9+s.header.stamp.nanosec))/1e9
    if not 0<=age<1.:raise RuntimeError('Published robot state stale')
    print(json.dumps(dict(O_T_EE=pose(s.o_t_ee),F_T_EE=pose(s.f_t_ee),q=list(s.measured_joint_state.position),robot_mode=int(s.robot_mode),source='existing_ROS_robot_state_no_second_FCI_connection',host_monotonic_before=start,host_monotonic_after=time.monotonic())))
finally:n.destroy_node();rclpy.shutdown()
