from .Native import ffi, native
from .DeviceViewable import DeviceViewable
import ctypes

class DVBufferLike(DeviceViewable):
    def size(self):
        return native.n_dvbufferlike_size(self.m_cptr)

    def from_host(self, ptr_host_data):
        native.n_dvbufferlike_from_host(self.m_cptr, ffi.cast("void *", ptr_host_data))

    def to_host(self, ptr_host_data, begin = 0, end = -1):
        native.n_dvbufferlike_to_host(self.m_cptr, ffi.cast("void *", ptr_host_data), begin, end)

    def range(self, begin = 0, end = -1):
        return DVBufferRange(self, begin, end)

class DVBuffer(DVBufferLike):
    def __init__(self, cptr):
        self.m_cptr = cptr

class DVBufferRange(DVBufferLike):
    def __init__(self, src, begin = 0, end = -1):
        self.m_src = src
        self.m_cptr = native.n_dvbuffer_range_from_dvbuffer(src.m_cptr, ctypes.c_ulonglong(begin).value, ctypes.c_ulonglong(end).value)

def device_buffer(size, ptr_host_data=None):
    ffiptr = ffi.NULL
    if ptr_host_data!=None:
        ffiptr = ffi.cast("void *", ptr_host_data)
    return DVBuffer(native.n_dvbuffer_create(size, ffiptr))

def device_buffer_from_dvs(lst_dv):
    dvarr = ObjArray(lst_dv)
    return DVBuffer(native.n_dvbuffer_from_dvs(dvarr.m_cptr))

class DVNumbaBuffer(DVBufferLike):
    def __init__(self, nbarr, size):
        self.nbarr = nbarr
        ptr_device_data = nbarr.device_ctypes_pointer.value
        self.m_cptr = native.n_dvbuffer_range_create(size, ffi.cast("void *", ptr_device_data))

