"""Offline C++ regression for the exact patched RAII class using fake schedulers.

No Robot construction, ROS node, realtime policy mutation or hardware connection.
"""
from pathlib import Path
import subprocess,tempfile

source=Path('/home/nshenoy/franka_ros2_ws/src/franka_ros2/franka_hardware/src/robot.cpp').read_text()
start=source.index('class ConstructorSchedulerRestore {');end=source.index('\n};',start)+3
cpp=r'''
#include <pthread.h>
#include <cstdio>
#include <cstring>
#include <system_error>
#include <stdexcept>
#include <cassert>
int policy=SCHED_OTHER,priority=0,get_error=0,set_failures=0;
int fake_get(pthread_t,int* p,sched_param* s){if(get_error)return get_error;*p=policy;s->sched_priority=priority;return 0;}
int fake_set(pthread_t,int p,const sched_param* s){if(set_failures){--set_failures;return EPERM;}policy=p;priority=s->sched_priority;return 0;}
#define pthread_getschedparam fake_get
#define pthread_setschedparam fake_set
'''+source[start:end]+r'''
int main(){
 {ConstructorSchedulerRestore guard;policy=SCHED_FIFO;priority=99;guard.restore();assert(policy==SCHED_OTHER&&priority==0);}
 try{ConstructorSchedulerRestore guard;policy=SCHED_FIFO;priority=99;throw std::runtime_error("simulated libfranka constructor failure");}catch(const std::runtime_error&){}
 assert(policy==SCHED_OTHER&&priority==0);
 bool failed=false;
 try{ConstructorSchedulerRestore guard;policy=SCHED_FIFO;priority=99;set_failures=1;guard.restore();}catch(const std::system_error&){failed=true;}
 assert(failed&&policy==SCHED_OTHER&&priority==0);
 policy=SCHED_FIFO;priority=50;
 {ConstructorSchedulerRestore guard;priority=99;guard.restore();}
 assert(policy==SCHED_FIFO&&priority==50);
 get_error=EACCES;failed=false;
 try{ConstructorSchedulerRestore guard;}catch(const std::system_error&){failed=true;}
 assert(failed&&policy==SCHED_FIFO&&priority==50);
 std::puts("PASS: normal restore, constructor-exception rollback, restore-failure rollback, incoming FIFO50 preservation, save failure");
}
'''
with tempfile.TemporaryDirectory()as directory:
    code=Path(directory)/'test.cpp';binary=Path(directory)/'test';code.write_text(cpp)
    subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-pthread',str(code),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)
