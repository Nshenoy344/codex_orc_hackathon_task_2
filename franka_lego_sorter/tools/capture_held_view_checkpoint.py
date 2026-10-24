"""Read ROS TCP/Hand state and existing SDK capture; sends no actions/camera request."""
import argparse,json,time
from pathlib import Path
from lego_sorter.cli import load_config
from lego_sorter.motion import RosRobot
from lego_sorter.held_view_checkpoint import build_checkpoint


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot',required=True);p.add_argument('--result',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    if out.exists():raise RuntimeError('Existing checkpoint must not be overwritten')
    r=RosRobot(load_config('config/demo.yaml'))
    try:
        end=time.monotonic()+6
        while True:
            try:actual=r.current_pose();break
            except RuntimeError:
                if time.monotonic()>end:raise
                time.sleep(.05)
        c=build_checkpoint(a.snapshot,a.result,actual,r.gripper_state());out.write_text(json.dumps(c,indent=2));print(json.dumps(c,indent=2))
    finally:r.close()

if __name__=='__main__':main()
