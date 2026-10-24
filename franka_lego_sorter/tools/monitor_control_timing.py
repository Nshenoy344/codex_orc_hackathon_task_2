"""Read-only, bounded-rate ROS observation of real FCI command success."""
import argparse,csv,json,math,time
from pathlib import Path
import rclpy
from rclpy.qos import QoSProfile,ReliabilityPolicy,HistoryPolicy
from franka_msgs.msg import FrankaRobotState
p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=300);p.add_argument('--output',required=True);a=p.parse_args()
rclpy.init();node=rclpy.create_node('lego_control_timing_monitor');rows=[];start=time.monotonic();last=None;last_report=start;max_gap=0.;faults=set()
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
def callback(msg):
 global last,max_gap
 now=time.monotonic()
 if last is not None:max_gap=max(max_gap,now-last)
 last=now;rate=float(msg.control_command_success_rate)
 rows.append((now-start,rate,int(msg.robot_mode)))
 for name in msg.current_errors.get_fields_and_field_types():
  if getattr(msg.current_errors,name):faults.add(name)
qos=QoSProfile(history=HistoryPolicy.KEEP_LAST,depth=1,reliability=ReliabilityPolicy.BEST_EFFORT)
sub=node.create_subscription(FrankaRobotState,'/franka_robot_state_broadcaster/robot_state',callback,qos)
def report(terminal=False,reason=None):
 rates=[r[1]for r in rows if math.isfinite(r[1])]
 data={'read_only':True,'terminal':terminal,'elapsed_sec':time.monotonic()-start,'sample_count':len(rows),'minimum_command_success_rate':min(rates)if rates else None,'mean_command_success_rate':sum(rates)/len(rates)if rates else None,'maximum_observed_message_gap_sec':max_gap,'last_message_age_sec':time.monotonic()-last if last else None,'robot_errors':sorted(faults),'reason':reason,'motion_command_sent':False}
 out.write_text(json.dumps(data,indent=2));print(json.dumps(data),flush=True)
try:
 while rclpy.ok()and time.monotonic()-start<a.seconds:
  rclpy.spin_once(node,timeout_sec=.02);time.sleep(.01)
  if time.monotonic()-last_report>=1:
   report();last_report=time.monotonic()
  if (last and time.monotonic()-last>1.)or(not last and time.monotonic()-start>10.):raise RuntimeError('Robot-state stream absent/stale')
 report(True)
except BaseException as e:
 report(True,str(e));raise
finally:
 with out.with_suffix('.csv').open('w')as f:
  writer=csv.writer(f);writer.writerow(['elapsed_sec','control_command_success_rate','robot_mode']);writer.writerows(rows)
 node.destroy_node();rclpy.shutdown()
