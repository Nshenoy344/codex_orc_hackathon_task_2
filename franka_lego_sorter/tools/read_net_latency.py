#!/usr/bin/env python3
"""Read only Linux ETHTOOL_GEEE/GCOALESCE ioctls; sends no network packets."""
import ctypes,fcntl,json,socket


def read(command,names):
    buffer=(ctypes.c_uint32*len(names))();buffer[0]=command
    # struct ifreq:16byte interface name,24byte union containing data pointer.
    class Ifreq(ctypes.Structure):
        _fields_=[('name',ctypes.c_char*16),('data',ctypes.c_void_p),('padding',ctypes.c_char*16)]
    request=Ifreq();request.name=b'eno1';request.data=ctypes.addressof(buffer)
    try:
        with socket.socket(socket.AF_INET,socket.SOCK_DGRAM)as s:fcntl.ioctl(s.fileno(),0x8946,request)
        return dict(zip(names,map(int,buffer)))
    except OSError as exc:return {'error':str(exc)}


a=['cmd','supported','advertised','lp_advertised','eee_active','eee_enabled','tx_lpi_enabled','tx_lpi_timer','reserved0','reserved1']
b=['cmd','rx_coalesce_usecs','rx_max_coalesced_frames','rx_coalesce_usecs_irq','rx_max_coalesced_frames_irq','tx_coalesce_usecs','tx_max_coalesced_frames','tx_coalesce_usecs_irq','tx_max_coalesced_frames_irq','stats_block_coalesce_usecs','use_adaptive_rx_coalesce','use_adaptive_tx_coalesce','pkt_rate_low','rx_coalesce_usecs_low','rx_max_coalesced_frames_low','tx_coalesce_usecs_low','tx_max_coalesced_frames_low','pkt_rate_high','rx_coalesce_usecs_high','rx_max_coalesced_frames_high','tx_coalesce_usecs_high','tx_max_coalesced_frames_high','rate_sample_interval']
print(json.dumps({'interface':'eno1','read_only':True,'eee':read(0x44,a),'coalesce':read(0x0e,b)},indent=2))
