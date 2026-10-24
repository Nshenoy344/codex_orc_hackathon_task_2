"""Remove confirmed libfranka-constructor FIFO99 leakage from ROS support threads.

This host's explicit controller update thread uses FIFO50. No robot command,
affinity change or update-thread scheduling change is performed. Default: inspect.
"""
import argparse,json,os,time
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--pid',type=int,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--apply',action='store_true')
a=p.parse_args()
if a.output.exists():raise RuntimeError('Use a unique audit file')
proc=Path('/proc')/str(a.pid)
if proc.joinpath('exe').resolve()!=Path('/opt/ros/humble/lib/controller_manager/ros2_control_node'):
    raise RuntimeError('Expected exact ROS controller executable')
def scan():
    rows=[]
    for t in sorted(proc.joinpath('task').iterdir(),key=lambda q:int(q.name)):
        try:
            tid=int(t.name);policy=os.sched_getscheduler(tid);priority=os.sched_getparam(tid).sched_priority
            rows.append(dict(tid=tid,name=t.joinpath('comm').read_text().strip(),policy=policy,priority=priority,starttime=t.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]))
        except (FileNotFoundError,ProcessLookupError):continue
    return rows
before=scan()
if sum(v['policy']==os.SCHED_FIFO and v['priority']==50 for v in before)!=1:
    raise RuntimeError('Expected one explicit FIFO50 update thread; inspect manually')
report=dict(pid=a.pid,apply=a.apply,time=time.time(),before=before,changes=[],robot_command_sent=False)
a.output.write_text(json.dumps(report,indent=2))
if a.apply:
    for repeat in range(3):
        rows=scan()
        for row in sorted(rows,key=lambda v:v['tid']!=a.pid):
            if row['policy']==os.SCHED_FIFO and row['priority']==99:
                os.sched_setscheduler(row['tid'],os.SCHED_OTHER,os.sched_param(0))
                report['changes'].append(row)
                a.output.write_text(json.dumps(report,indent=2))
        time.sleep(.02)
report['after']=scan()
report['update_fifo50_preserved']=any(v['policy']==os.SCHED_FIFO and v['priority']==50 for v in report['after'])
report['fifo99_remaining']=sum(v['policy']==os.SCHED_FIFO and v['priority']==99 for v in report['after'])
a.output.write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items()if k not in('before','after','changes')},indent=2))
print('Changed support threads:',len(report['changes']))
if not report['update_fifo50_preserved']or(a.apply and report['fifo99_remaining']):raise RuntimeError('Scheduling readback failed')
