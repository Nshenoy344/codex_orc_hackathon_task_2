"""Temporary CPU/NIC latency guard; no robot control connections.

Root privileges required. Stop robot control before stopping this guard:
SIGINT/SIGTERM restores original CPU governors and EEE settings, and closes
/dev/cpu_dma_latency. The state report is not a motion permit.
"""
import argparse,json,os,signal,struct,threading
from pathlib import Path
from host_network_io import eee_ioctl,disabled_eee


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state-file',required=True)
    p.add_argument('--interface',default='eno1');p.add_argument('--disable-eee',action='store_true');a=p.parse_args()
    if os.geteuid()!=0:raise SystemExit('Root required for CPU governor, latency device and requested EEE setting')
    stop=threading.Event();signal.signal(signal.SIGINT,lambda *_:stop.set());signal.signal(signal.SIGTERM,lambda *_:stop.set())
    original={};fd=None;original_eee=None;eee_restore_needed=False;state=Path(a.state_file);error=None;restore_errors=[]
    def report(active):
        data={'pid':os.getpid(),'active':active,'latency_request_us':0,'original_governors':original,
            'current_governors':{name:Path(name).read_text().strip()for name in original},'error':error,'restore_errors':restore_errors,
            'robot_connection_opened':False,'temporary_only':True,'interface':a.interface,'disable_eee_requested':a.disable_eee,
            'original_eee':original_eee,'current_eee':eee_ioctl(a.interface)if original_eee is not None else None}
        state.write_text(json.dumps(data,indent=2));print(json.dumps({'pid':data['pid'],'active':active,'error':error,'restore_errors':restore_errors}),flush=True)
    try:
        # Read complete original NIC state before any mutation for exact restoration.
        if a.disable_eee:original_eee=eee_ioctl(a.interface)
        for governor in sorted(Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_governor')):
            if 'performance'not in(governor.parent/'scaling_available_governors').read_text().split():raise RuntimeError('Performance governor unavailable: '+str(governor))
            original[str(governor)]=governor.read_text().strip();governor.write_text('performance\n')
        if not original:raise RuntimeError('No CPU scaling policies found')
        fd=os.open('/dev/cpu_dma_latency',os.O_RDWR);os.write(fd,struct.pack('i',0))
        if a.disable_eee:
            eee_restore_needed=True # Restore even if SET raises after a partial device update.
            eee_ioctl(a.interface,disabled_eee(original_eee))
            if eee_ioctl(a.interface)['eee_enabled']:raise RuntimeError('NIC EEE disable did not take effect')
        report(True)
        while not stop.wait(.5):pass
    except BaseException as exc:
        error=str(exc)
        raise
    finally:
        if eee_restore_needed:
            try:eee_ioctl(a.interface,original_eee)
            except Exception as exc:restore_errors.append('EEE restoration failed: '+str(exc))
        if fd is not None:
            try:os.close(fd)
            except Exception as exc:restore_errors.append('Latency FD close failed: '+str(exc))
        for filename,value in original.items():
            try:Path(filename).write_text(value+'\n')
            except Exception as exc:restore_errors.append('Governor restoration failed '+filename+': '+str(exc))
        report(False)
    if restore_errors:raise SystemExit('Some original host settings could not be restored; inspect state file')

if __name__=='__main__':main()
