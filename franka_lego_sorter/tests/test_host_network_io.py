import ctypes,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from host_network_io import EEE_FIELDS,eee_ioctl,disabled_eee

class Socket:
    def __enter__(self):return self
    def __exit__(self,*_):pass
    def fileno(self):return 7

class HostNetworkTests(unittest.TestCase):
    def test_get_disable_restore_full_original(self):
        original=[0x44,40,40,40,1,1,1,27,0,0];calls=[]
        def ioctl(fd,command,req):
            self.assertEqual((fd,command,req.name),(7,0x8946,b'eno1'))
            fields=(ctypes.c_uint32*10).from_address(req.data);calls.append(list(fields))
            if fields[0]==0x44:
                for i,v in enumerate(original):fields[i]=v
        saved=eee_ioctl('eno1',ioctl=ioctl,socket_factory=lambda *_:Socket())
        off=disabled_eee(saved);eee_ioctl('eno1',off,ioctl=ioctl,socket_factory=lambda *_:Socket())
        eee_ioctl('eno1',saved,ioctl=ioctl,socket_factory=lambda *_:Socket())
        self.assertEqual(calls[1][0],0x45);self.assertEqual(calls[1][5:8],[0,0,27])
        self.assertEqual(calls[2],[0x45,*original[1:]])
        self.assertEqual(saved['eee_enabled'],1)
    def test_refuses_incomplete_restore(self):
        with self.assertRaisesRegex(ValueError,'every original'):eee_ioctl('eno1',{'eee_enabled':0})
    def test_refuses_invalid_interface(self):
        with self.assertRaises(ValueError):eee_ioctl('a'*16)
