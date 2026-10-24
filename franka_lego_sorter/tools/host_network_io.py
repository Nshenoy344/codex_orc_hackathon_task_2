"""Linux ethtool EEE GET/SET ABI; no packets or robot control connections."""
import ctypes,fcntl,socket

EEE_FIELDS=('cmd','supported','advertised','lp_advertised','eee_active','eee_enabled','tx_lpi_enabled','tx_lpi_timer','reserved0','reserved1')

class Ifreq(ctypes.Structure):
    _fields_=[('name',ctypes.c_char*16),('data',ctypes.c_void_p),('padding',ctypes.c_char*16)]


def eee_ioctl(interface,values=None,ioctl=fcntl.ioctl,socket_factory=socket.socket):
    name=interface.encode('ascii')
    if not 0<len(name)<16 or b'\0'in name:raise ValueError('Invalid Linux interface name')
    if values is not None and set(values)!=set(EEE_FIELDS):raise ValueError('EEE restoration requires every original ABI field')
    buffer=(ctypes.c_uint32*10)(*(int(values[k])for k in EEE_FIELDS))if values is not None else(ctypes.c_uint32*10)()
    buffer[0]=0x45 if values is not None else 0x44 # ETHTOOL_SEEE / ETHTOOL_GEEE
    request=Ifreq();request.name=name;request.data=ctypes.addressof(buffer)
    with socket_factory(socket.AF_INET,socket.SOCK_DGRAM)as sock:ioctl(sock.fileno(),0x8946,request)
    return dict(zip(EEE_FIELDS,map(int,buffer)))


def disabled_eee(original):
    values=dict(original);values['eee_enabled']=0;values['tx_lpi_enabled']=0
    return values
